import math
import traceback
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from numpy import ndarray
from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[tuple[int, int]]


class Miner:
    input_size = 640
    num_queries = 300
    interp = cv2.INTER_AREA
    use_cuda_graph = False

    class_names = ["road_sign"]

    iou_thres = 0.5
    cross_iou_thresh = 0.8
    max_det = 150

    _conf_thres_array = np.array([0.40], dtype=np.float32)
    _bonus_array = np.array([0.05], dtype=np.float32)

    min_box_area = 9
    min_side = 3
    max_aspect_ratio = 12.0

    tile_trigger_ratio = 1.4
    tile_overlap_ratio = 0.20

    def __init__(self, path_hf_repo: Path) -> None:
        model_path = Path(path_hf_repo) / "weights.onnx"

        try:
            ort.preload_dlls()
        except Exception as e:
            print(f"miner init: preload_dlls failed: {e}", flush=True)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        providers = self._provider_list_with_opts()
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                disabled_optimizers=["SimplifiedLayerNormFusion"],
                providers=providers,
            )
        except Exception as e:
            print(
                f"miner init: session creation failed, retrying without disabled optimizers: {e}",
                flush=True,
            )
            try:
                self.session = ort.InferenceSession(
                    str(model_path), sess_options=sess_options, providers=providers
                )
            except Exception as e2:
                print(f"miner init: session creation failed, falling back to CPU: {e2}", flush=True)
                self.session = ort.InferenceSession(
                    str(model_path),
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )

        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self._boxes_name = self.output_names[1]
        self._scores_name = self.output_names[2]

        self._init_buffers()
        self._on_cuda = self.session.get_providers()[0] == "CUDAExecutionProvider"
        self._init_iobinding()
        self._warmup()

        self.use_tta = False
        self.use_tile_tta = False
        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_score_thresh = 0.01

        active = self.session.get_providers()[0]
        print(f"Model loaded from {model_path}  provider={active}  iobinding=ON")
        print("per-class conf: " + ", ".join(
            f"{n}={t:.3f}" for n, t in zip(
                self.class_names, self._conf_thres_array.tolist()
            )
        ))

    @staticmethod
    def _providers() -> list[str]:
        avail = ort.get_available_providers()
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]

    def _provider_list_with_opts(self):
        cuda_opts: dict[str, str] = {"cudnn_conv_algo_search": "HEURISTIC"}
        if self.use_cuda_graph:
            cuda_opts["enable_cuda_graph"] = "1"
        out = []
        for p in self._providers():
            if p == "CUDAExecutionProvider":
                out.append((p, cuda_opts))
            else:
                out.append(p)
        return out

    def _init_buffers(self) -> None:
        size = self.input_size
        q = self.num_queries

        self._canvas = np.zeros((size, size, 3), dtype=np.uint8)
        self._rgb = np.empty((size, size, 3), dtype=np.uint8)
        self._input_tensor = np.zeros((1, 3, size, size), dtype=np.float32)
        self._sizes = np.array([[size, size]], dtype=np.int64)

        self._out_labels = np.empty((1, q), dtype=np.int64)
        self._out_boxes = np.empty((1, q, 4), dtype=np.float32)
        self._out_scores = np.empty((1, q), dtype=np.float32)

    def _init_iobinding(self) -> None:
        self._io_binding = self.session.io_binding()
        self._io_binding.bind_cpu_input(self.input_name, self._input_tensor)
        self._io_binding.bind_cpu_input(self.size_name, self._sizes)
        if self._on_cuda:
            for name in self.output_names:
                self._io_binding.bind_output(name, "cuda")
        else:
            for name, buf, dtype in (
                (self.output_names[0], self._out_labels, np.int64),
                (self._boxes_name, self._out_boxes, np.float32),
                (self._scores_name, self._out_scores, np.float32),
            ):
                self._io_binding.bind_output(
                    name, "cpu", 0, dtype, buf.shape, buf.ctypes.data
                )

    def _warmup(self) -> None:
        self._input_tensor.fill(0.0)
        if self._on_cuda:
            self.session.run(
                self.output_names,
                {self.input_name: self._input_tensor, self.size_name: self._sizes},
            )
        else:
            self.session.run_with_iobinding(self._io_binding)

    def _letterbox(self, image: ndarray) -> tuple[float, int, int]:
        size = self.input_size
        oh, ow = image.shape[:2]
        ratio = min(size / ow, size / oh)
        nw, nh = int(ow * ratio), int(oh * ratio)

        resized = cv2.resize(image, (nw, nh), interpolation=self.interp)
        pad_w, pad_h = (size - nw) // 2, (size - nh) // 2

        self._canvas.fill(0)
        self._canvas[pad_h: pad_h + nh, pad_w: pad_w + nw] = resized
        cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB, dst=self._rgb)

        chw = self._input_tensor[0]
        np.copyto(
            chw,
            self._rgb.transpose(2, 0, 1).astype(np.float32, copy=False) / 255.0,
        )
        return ratio, pad_w, pad_h

    def _run_inference(self) -> None:
        if self._on_cuda:
            _, boxes, scores = self.session.run(
                self.output_names,
                {self.input_name: self._input_tensor, self.size_name: self._sizes},
            )
            np.copyto(self._out_boxes, boxes)
            np.copyto(self._out_scores, scores)
            return

        self.session.run_with_iobinding(self._io_binding)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        w, h = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _hard_nms(
        boxes: np.ndarray, scores: np.ndarray, iou_thresh: float
    ) -> np.ndarray:
        n = len(boxes)
        if n == 0:
            return np.array([], dtype=np.intp)
        order = np.argsort(-scores)
        keep: list[int] = []
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = (max(0.0, boxes[i, 2] - boxes[i, 0]) *
                   max(0.0, boxes[i, 3] - boxes[i, 1]))
            a_r = (np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) *
                   np.maximum(0.0, boxes[rest, 3] - boxes[rest, 1]))
            iou = inter / (a_i + a_r - inter + 1e-7)
            order = rest[iou <= iou_thresh]
        return np.array(keep, dtype=np.intp)

    def _per_class_hard_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        iou_thresh: float,
    ) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=np.intp)
        all_keep: list[int] = []
        for c in np.unique(cls_ids):
            mask = cls_ids == c
            indices = np.where(mask)[0]
            keep = self._hard_nms(boxes[mask], scores[mask], iou_thresh)
            all_keep.extend(indices[keep].tolist())
        all_keep.sort()
        return np.array(all_keep, dtype=np.intp)

    def _soft_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        sigma: float = 0.5,
        score_thresh: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        N = len(boxes)
        if N == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        boxes = boxes.astype(np.float32, copy=True)
        scores = scores.astype(np.float32, copy=True)
        order = np.arange(N)
        for i in range(N):
            max_pos = i + int(np.argmax(scores[i:]))
            boxes[[i, max_pos]] = boxes[[max_pos, i]]
            scores[[i, max_pos]] = scores[[max_pos, i]]
            order[[i, max_pos]] = order[[max_pos, i]]
            if i + 1 >= N:
                break
            xx1 = np.maximum(boxes[i, 0], boxes[i + 1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i + 1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i + 1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i + 1:, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_i = max(0.0, float(
                (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])))
            areas_j = (np.maximum(0.0, boxes[i + 1:, 2] - boxes[i + 1:, 0])
                       * np.maximum(0.0, boxes[i + 1:, 3] - boxes[i + 1:, 1]))
            iou = inter / (area_i + areas_j - inter + 1e-7)
            scores[i + 1:] *= np.exp(-(iou ** 2) / sigma)
        mask = scores > score_thresh
        return order[mask], scores[mask]

    def _per_class_soft_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        sigma: float = 0.5,
        score_thresh: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        all_keep: list[int] = []
        all_scores: list[float] = []
        for c in np.unique(cls_ids):
            indices = np.where(cls_ids == c)[0]
            keep, updated = self._soft_nms(boxes[indices], scores[indices],
                                           sigma, score_thresh)
            for k, s in zip(keep, updated):
                all_keep.append(int(indices[k]))
                all_scores.append(float(s))
        if not all_keep:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        return np.array(all_keep, dtype=np.intp), np.array(all_scores, dtype=np.float32)

    def _cross_class_dedup_op(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        iou_thresh: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        cls_ids = np.asarray(cls_ids, dtype=np.int32)
        areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) *
                 np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
        margins = scores - self._conf_thres_array[cls_ids]
        order = np.lexsort((-areas, -margins))
        suppressed = np.zeros(n, dtype=bool)
        keep: list[int] = []
        for i in order:
            if suppressed[i]:
                continue
            keep.append(int(i))
            bi = boxes[i]
            xx1 = np.maximum(bi[0], boxes[:, 0])
            yy1 = np.maximum(bi[1], boxes[:, 1])
            xx2 = np.minimum(bi[2], boxes[:, 2])
            yy2 = np.minimum(bi[3], boxes[:, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = max(1e-7, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
            iou = inter / (a_i + areas - inter + 1e-7)
            dup = iou > iou_thresh
            dup[i] = False
            suppressed |= dup
        keep_idx = np.array(keep, dtype=np.intp)
        return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]

    @staticmethod
    def _max_score_per_cluster(
        post_boxes: np.ndarray,
        post_cls: np.ndarray,
        full_boxes: np.ndarray,
        full_scores: np.ndarray,
        full_cls: np.ndarray,
        iou_thresh: float,
    ) -> np.ndarray:
        n = len(post_boxes)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        full_areas = (np.maximum(0.0, full_boxes[:, 2] - full_boxes[:, 0]) *
                      np.maximum(0.0, full_boxes[:, 3] - full_boxes[:, 1]))
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            bi = post_boxes[i]
            xx1 = np.maximum(bi[0], full_boxes[:, 0])
            yy1 = np.maximum(bi[1], full_boxes[:, 1])
            xx2 = np.minimum(bi[2], full_boxes[:, 2])
            yy2 = np.minimum(bi[3], full_boxes[:, 3])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            a_i = max(0.0, float((bi[2] - bi[0]) * (bi[3] - bi[1])))
            iou = inter / (a_i + full_areas - inter + 1e-7)
            cluster = (iou >= iou_thresh) & (full_cls == post_cls[i])
            out[i] = float(np.max(full_scores[cluster])) if np.any(cluster) else 0.0
        return out

    def _conf_filter_mask(
        self, scores: np.ndarray, cls_ids: np.ndarray
    ) -> np.ndarray:
        if len(scores) == 0:
            return np.zeros(0, dtype=bool)
        thr = self._conf_thres_array[cls_ids]
        keep = scores >= thr
        for c in np.unique(cls_ids):
            b = float(self._bonus_array[c])
            if b <= 0.0:
                continue
            cm = cls_ids == c
            if keep[cm].any():
                continue
            idx = np.where(cm)[0]
            top = int(idx[int(np.argmax(scores[idx]))])
            if scores[top] >= self._conf_thres_array[c] - b:
                keep[top] = True
        return keep

    def _filter_sane_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        orig_size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        orig_w, orig_h = orig_size
        image_area = float(orig_w * orig_h)
        keep = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if bw < self.min_side or bh < self.min_side:
                continue
            area = bw * bh
            if area < self.min_box_area:
                continue
            if area > 0.95 * image_area:
                continue
            ar = max(bw / max(bh, 1e-6), bh / max(bw, 1e-6))
            if ar > self.max_aspect_ratio:
                continue
            keep.append(i)
        if not keep:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )
        k = np.array(keep, dtype=np.intp)
        return boxes[k], scores[k], cls_ids[k]

    def _per_view_pipeline(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(boxes) > 1:
            if self.use_soft_nms:
                keep, new_scores = self._per_class_soft_nms(
                    boxes, scores, cls_ids,
                    self.soft_nms_sigma, self.soft_nms_score_thresh)
                boxes, scores, cls_ids = boxes[keep], new_scores, cls_ids[keep]
            else:
                keep = self._per_class_hard_nms(boxes, scores, cls_ids, self.iou_thres)
                boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) > self.max_det:
            top = np.argsort(-scores)[: self.max_det]
            boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._cross_class_dedup_op(
                boxes, scores, cls_ids, self.cross_iou_thresh
            )
        return boxes, scores, cls_ids

    @staticmethod
    def _build_results(
        boxes: np.ndarray, scores: np.ndarray, cls_ids: np.ndarray
    ) -> list[BoundingBox]:
        results: list[BoundingBox] = []
        for box, conf, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                BoundingBox(
                    x1=int(math.floor(x1)),
                    y1=int(math.floor(y1)),
                    x2=int(math.ceil(x2)),
                    y2=int(math.ceil(y2)),
                    cls_id=int(cls_id),
                    conf=float(conf),
                )
            )
        return results

    def _decode(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
        orig_w: int,
        orig_h: int,
    ) -> list[BoundingBox]:
        boxes = boxes.astype(np.float32)
        scores = scores.astype(np.float32)
        cls_ids = np.zeros(len(scores), dtype=np.int32)

        keep = self._conf_filter_mask(scores, cls_ids)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if len(boxes) == 0:
            return []

        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes = self._clip_boxes(boxes, (orig_w, orig_h))

        boxes, scores, cls_ids = self._filter_sane_boxes(
            boxes, scores, cls_ids, (orig_w, orig_h)
        )
        if len(boxes) == 0:
            return []

        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _predict_single(self, image: ndarray) -> list[BoundingBox]:
        ratio, pad_w, pad_h = self._letterbox(image)
        self._run_inference()
        return self._decode(
            self._out_boxes[0],
            self._out_scores[0],
            ratio,
            pad_w,
            pad_h,
            image.shape[1],
            image.shape[0],
        )

    def _predict_tiles(self, image: np.ndarray) -> list[BoundingBox]:
        h, w = image.shape[:2]
        if w < int(self.input_size * self.tile_trigger_ratio):
            return []

        overlap = int(w * self.tile_overlap_ratio)
        mid = w // 2
        x_left_end = min(w, mid + overlap // 2)
        x_right_start = max(0, mid - overlap // 2)

        left = image[:, :x_left_end]
        right = image[:, x_right_start:]

        boxes_left = self._predict_single(left)
        boxes_right = self._predict_single(right)

        shifted_right = [
            BoundingBox(
                x1=b.x1 + x_right_start,
                y1=b.y1,
                x2=b.x2 + x_right_start,
                y2=b.y2,
                cls_id=b.cls_id,
                conf=b.conf,
            )
            for b in boxes_right
        ]
        return boxes_left + shifted_right

    def _merge_views(
        self,
        view_boxes: list[list[BoundingBox]],
        image_size: tuple[int, int],
    ) -> list[BoundingBox]:
        all_boxes: list[BoundingBox] = []
        for vb in view_boxes:
            all_boxes.extend(vb)
        if not all_boxes:
            return []

        coords = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in all_boxes], dtype=np.float32
        )
        scores = np.array([b.conf for b in all_boxes], dtype=np.float32)
        cls_ids = np.array([b.cls_id for b in all_boxes], dtype=np.int32)

        coords = self._clip_boxes(coords, image_size)

        hard_keep = self._per_class_hard_nms(coords, scores, cls_ids, self.iou_thres)
        if len(hard_keep) == 0:
            return []
        if len(hard_keep) > self.max_det:
            top = np.argsort(-scores[hard_keep])[: self.max_det]
            hard_keep = hard_keep[top]

        boosted = self._max_score_per_cluster(
            coords[hard_keep], cls_ids[hard_keep],
            coords, scores, cls_ids, self.iou_thres,
        )

        kept_coords = coords[hard_keep]
        kept_cls = cls_ids[hard_keep]
        if len(kept_coords) > 1:
            kept_coords, boosted, kept_cls = self._cross_class_dedup_op(
                kept_coords, boosted, kept_cls, self.cross_iou_thresh
            )

        return [
            BoundingBox(
                x1=int(math.floor(kept_coords[j, 0])),
                y1=int(math.floor(kept_coords[j, 1])),
                x2=int(math.ceil(kept_coords[j, 2])),
                y2=int(math.ceil(kept_coords[j, 3])),
                cls_id=int(kept_cls[j]),
                conf=float(boosted[j]),
            )
            for j in range(len(kept_coords))
        ]

    def _predict_full(self, image: np.ndarray) -> list[BoundingBox]:
        if not self.use_tta and not self.use_tile_tta:
            return self._predict_single(image)

        views: list[list[BoundingBox]] = []
        if self.use_tta:
            views.append(self._predict_single(image))
            flipped = cv2.flip(image, 1)
            w = image.shape[1]
            flipped_dets = self._predict_single(flipped)
            views.append([
                BoundingBox(
                    x1=w - b.x2, y1=b.y1, x2=w - b.x1, y2=b.y2,
                    cls_id=b.cls_id, conf=b.conf,
                )
                for b in flipped_dets
            ])
        else:
            views.append(self._predict_single(image))

        if self.use_tile_tta:
            tile_boxes = self._predict_tiles(image)
            if tile_boxes:
                views.append(tile_boxes)

        h, w = image.shape[:2]
        return self._merge_views(views, (w, h))

    def predict_batch(
        self,
        batch_images: list[ndarray],
        offset: int,
        n_keypoints: int,
    ) -> list[TVFrameResult]:
        results: list[TVFrameResult] = []
        kp = [(0, 0) for _ in range(max(0, int(n_keypoints)))]

        for i, image in enumerate(batch_images):
            frame_id = offset + i
            try:
                frame_boxes = self._predict_full(image)
            except Exception as e:
                print(
                    f"predict_batch: inference failed for frame_id={frame_id}: {e}",
                    flush=True,
                )
                traceback.print_exc()
                frame_boxes = []

            results.append(TVFrameResult(
                frame_id=frame_id,
                boxes=frame_boxes,
                keypoints=list(kp),
            ))
        return results
