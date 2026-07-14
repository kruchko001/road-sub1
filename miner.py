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
    """ONNX Runtime miner for road-sign detection (single class).
    Strategy (ported from offense / fire001 miner):
      - per-class confidence threshold with per-class rescue bonus
      - per-class hard NMS, then cross-class dedup (no-op for single class)
      - horizontal-flip TTA with full-set cluster score boost
    Plus: class remap, sanity-box filter tuned for small distant signs,
    TTA toggle.
    """

    class_names = ["road sign"]
    # Order the model emits classes in -- remapped to `class_names` index.
    _model_class_order = ["road sign"]

    iou_thres = 0.5
    cross_iou_thresh = 0.8
    max_det = 150

    # Per-class confidence threshold. Road signs in this dataset are
    # frequently degraded / rear-facing / partly-obscured / distant, so we
    # run noticeably below the fire/smoke baseline. The validator's
    # false_positive pillar = max(0, 1 - ffpi/10): we can tolerate ~2 FP per
    # image and still keep that pillar above 0.8.
    _conf_thres_array = np.array(
        [0.33], dtype=np.float32
    )
    # Per-class rescue bonus. If a class has ZERO boxes passing the threshold
    # in a frame, its top-1 candidate is admitted when its score is at least
    # (threshold - bonus). Bumped from 0.05 -> 0.10 so a single faint sign in
    # an otherwise empty frame still produces a detection (map50 recall win,
    # at most one extra FP per such frame).
    _bonus_array = np.array(
        [0.18], dtype=np.float32
    )

    # Box sanity filter: drop tiny / degenerate / image-spanning / extreme
    # aspect ratio boxes.
    #   min_box_area = 14x14  -> 14x14 is the smallest credible sign. The old
    #                          value of 64 (8x8) silently discarded narrow
    #                          distant signs like a 10x6 px overhead chevron.
    #   min_side     = 3      -> matches min_box_area; anything thinner is
    #                          almost certainly a pole or shadow false alarm.
    #   max_aspect_ratio = 12.0
    #                         -> overhead destination panels and lane-assignment
    #                          signs are very wide (long, thin rectangles);
    #                          8.0 was clipping legitimate detections.
    min_box_area = 8 * 8
    min_side = 3
    max_aspect_ratio = 12.0

    # Tile-based TTA: when the source image is significantly larger than the
    # model input, letterboxing throws away ~1.5x of effective resolution,
    # which kills small-sign recall. Splitting into overlapping horizontal
    # tiles preserves native resolution on each half. Triggered only when
    # source width >= tile_trigger_ratio * model_input_width to avoid wasted
    # compute on already-small images.
    tile_trigger_ratio = 1.4
    tile_overlap_ratio = 0.20

    def __init__(self, path_hf_repo: Path) -> None:
        model_path = path_hf_repo / "weights.onnx"
        print("ORT version:", ort.__version__)

        try:
            ort.preload_dlls()
            print("✅ onnxruntime.preload_dlls() success")
        except Exception as e:
            print(f"⚠️ preload_dlls failed: {e}")

        print("ORT available providers BEFORE session:", ort.get_available_providers())

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        print("ORT session providers:", self.session.get_providers())

        # Build cls_remap: for each model-emit index i,
        #   cls_remap[i] = self.class_names.index(model_class_order[i])
        # i.e. convert a model-side class id into the output class id that
        # downstream code (BoundingBox.cls_id, the per-class threshold/bonus
        # arrays) expects. The model-side order comes from the ONNX metadata
        # when available, else falls back to the static _model_class_order.
        model_class_order = self._read_model_class_order()
        if model_class_order is None:
            model_class_order = list(self._model_class_order)
            print(f"cls order: no usable ONNX metadata, FALLBACK {model_class_order}")
        else:
            print(f"cls order: from ONNX metadata {model_class_order}")
        self.cls_remap = np.array(
            [self.class_names.index(n) for n in model_class_order],
            dtype=np.int32,
        )

        for inp in self.session.get_inputs():
            print("INPUT:", inp.name, inp.shape, inp.type)
        for out in self.session.get_outputs():
            print("OUTPUT:", out.name, out.shape, out.type)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

        # weights.onnx is exported at 640x640 (Ultralytics imgsz metadata),
        # static (dynamic=False). The default is only the fallback for when the
        # ONNX input dims aren't fixed; the real value is read from the session.
        self.input_height = self._safe_dim(self.input_shape[2], default=640)
        self.input_width = self._safe_dim(self.input_shape[3], default=640)

        self.use_tta = False
        self.use_tile_tta = False
        # Soft-NMS (ported from carwash001): Gaussian score decay of overlapping
        # boxes instead of hard removal. OFF by default to preserve the current
        # deployed behaviour; flip on (and tune sigma) via tune_miner.py to see if
        # it scores better — useful where signs cluster (gantries, sign assemblies).
        self.use_soft_nms = False
        self.soft_nms_sigma = 0.5
        self.soft_nms_score_thresh = 0.01

        print(f"✅ ONNX model loaded from: {model_path}")
        print(f"✅ ONNX providers: {self.session.get_providers()}")
        print(f"✅ ONNX input: name={self.input_name}, shape={self.input_shape}")
        print(f"✅ ONNX input size: {self.input_width}x{self.input_height}, "
              f"use_tta={self.use_tta}, use_tile_tta={self.use_tile_tta}")
        print("per-class conf: " + ", ".join(
            f"{n}={t:.3f}" for n, t in zip(
                self.class_names, self._conf_thres_array.tolist()
            )
        ))

        self._warmup()

    def _warmup(self, iters: int = 3) -> None:
        try:
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            for _ in range(max(1, iters)):
                self.predict_batch(batch_images=[dummy], offset=0, n_keypoints=0)
            print(f"✅ warmup: {iters} dummy predict_batch call(s) done")
        except Exception as e:
            print(f"⚠️ warmup skipped: {e}")

    def _read_model_class_order(self) -> "list[str] | None":
        """Read the model's class order from Ultralytics ONNX metadata.
        Returns the class names ordered by model-emit index, or None when the
        metadata is missing/unparsable or doesn't match `class_names` as a set
        (in which case the static _model_class_order fallback is used)."""
        try:
            import ast

            meta = self.session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(meta["names"])  # e.g. {0: 'road_sign'}
            if isinstance(names, dict):
                order = [str(names[i]) for i in sorted(names)]
            else:
                order = [str(n) for n in names]
        except Exception as e:
            print(f"cls order: could not read ONNX names metadata ({e})")
            return None
        if sorted(order) != sorted(self.class_names):
            print(
                f"cls order: ONNX names {order} do not match expected classes "
                f"{self.class_names}; ignoring metadata"
            )
            return None
        return order

    def __repr__(self) -> str:
        return (
            f"ONNXRuntime(session={type(self.session).__name__}, "
            f"providers={self.session.get_providers()})"
        )

    @staticmethod
    def _safe_dim(value, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default

    def _letterbox(
        self,
        image: ndarray,
        new_shape: tuple[int, int],
        color=(114, 114, 114),
    ) -> tuple[ndarray, float, tuple[float, float]]:
        h, w = image.shape[:2]
        new_w, new_h = new_shape

        ratio = min(new_w / w, new_h / h)
        resized_w = int(round(w * ratio))
        resized_h = int(round(h * ratio))

        if (resized_w, resized_h) != (w, h):
            interp = cv2.INTER_CUBIC if ratio > 1.0 else cv2.INTER_LINEAR
            image = cv2.resize(image, (resized_w, resized_h), interpolation=interp)

        dw = (new_w - resized_w) / 2.0
        dh = (new_h - resized_h) / 2.0

        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))

        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=color,
        )
        return padded, ratio, (dw, dh)

    def _preprocess(
        self, image: ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        orig_h, orig_w = image.shape[:2]
        img, ratio, pad = self._letterbox(
            image, (self.input_width, self.input_height)
        )
        # Fused scale(1/255) + BGR->RGB swap + HWC->NCHW + contiguous float32 in
        # one optimized OpenCV call (bit-identical to the cvtColor + astype/255 +
        # transpose chain, but ~half the preprocess time).
        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0 / 255.0, swapRB=True)
        return blob, ratio, pad, (orig_w, orig_h)

    @staticmethod
    def _clip_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        w, h = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = np.empty_like(boxes)
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        return out

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
        """Soft-NMS: Gaussian decay of overlapping scores instead of hard removal.
        Returns (kept_original_indices, updated_scores). (Ported from carwash001.)"""
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
        """Soft-NMS applied independently per class. Returns (kept_idx, updated_scores)."""
        if len(boxes) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.float32)
        all_keep: list[int] = []
        all_scores: list[float] = []
        for c in np.unique(cls_ids):
            indices = np.where(cls_ids == c)[0]
            keep, updated = self._soft_nms(boxes[indices], scores[indices],
                                           sigma, score_thresh)
            for k, s in zip(keep, updated):
                all_keep.append(int(indices[k])); all_scores.append(float(s))
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
        """Remove near-duplicate boxes across classes.
        Order candidates by (score - per_class_threshold) margin, then by area;
        keep the highest, suppress every other box with IoU > iou_thresh.
        With a single road_sign class this is effectively a no-op, but the
        method is kept so the pipeline stays compatible with the multi-class
        miner template.
        """
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
        """For each kept (post-NMS) box, return the max score over the FULL
        candidate set among same-class boxes with IoU >= iou_thresh.
        Used after horizontal-flip TTA: a high-confidence flipped detection
        can raise the score of the corresponding original detection.
        """
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
        """Boolean keep-mask: score >= per-class threshold, with a per-class
        rescue -- if a class has zero boxes passing, admit its top-1 candidate
        when its score >= (per-class threshold - per-class bonus)."""
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
        """Drop tiny / degenerate / image-spanning / extreme-AR boxes (FP)."""
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
        """Per-view post-processing pipeline: per-class NMS -> cap -> cross-class dedup."""
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
                    x1=int(round(x1)),
                    y1=int(round(y1)),
                    x2=int(round(x2)),
                    y2=int(round(y2)),
                    cls_id=int(cls_id),
                    conf=float(conf),
                )
            )
        return results

    def _decode_final_dets(
        self,
        preds: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_size: tuple[int, int],
    ) -> list[BoundingBox]:
        """Final-detection output path: rows shaped [x1, y1, x2, y2, conf, cls_id]."""
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        if preds.ndim != 2 or preds.shape[1] < 6:
            raise ValueError(f"Unexpected ONNX final-det output shape: {preds.shape}")

        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)
        cls_ids = self.cls_remap[cls_ids]

        keep = self._conf_filter_mask(scores, cls_ids)
        boxes = boxes[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if len(boxes) == 0:
            return []

        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)

        boxes, scores, cls_ids = self._filter_sane_boxes(
            boxes, scores, cls_ids, orig_size
        )
        if len(boxes) == 0:
            return []

        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _decode_raw_yolo(
        self,
        preds: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_size: tuple[int, int],
    ) -> list[BoundingBox]:
        """Fallback raw-YOLO output path: per-anchor class logits."""
        if preds.ndim != 3 or preds.shape[0] != 1:
            raise ValueError(f"Unexpected raw ONNX output shape: {preds.shape}")
        preds = preds[0]
        if preds.shape[0] <= 16 and preds.shape[1] > preds.shape[0]:
            preds = preds.T
        if preds.ndim != 2 or preds.shape[1] < 5:
            raise ValueError(f"Unexpected raw output shape: {preds.shape}")

        boxes_xywh = preds[:, :4].astype(np.float32)
        cls_part = preds[:, 4:].astype(np.float32)
        if cls_part.shape[1] == 1:
            scores = cls_part[:, 0]
            cls_ids = np.zeros(len(scores), dtype=np.int32)
        else:
            cls_ids = np.argmax(cls_part, axis=1).astype(np.int32)
            scores = cls_part[np.arange(len(cls_part)), cls_ids]
        cls_ids = self.cls_remap[cls_ids]

        keep = self._conf_filter_mask(scores, cls_ids)
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        if len(boxes_xywh) == 0:
            return []
        boxes = self._xywh_to_xyxy(boxes_xywh)

        pad_w, pad_h = pad
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h
        boxes /= ratio
        boxes = self._clip_boxes(boxes, orig_size)

        boxes, scores, cls_ids = self._filter_sane_boxes(
            boxes, scores, cls_ids, orig_size
        )
        if len(boxes) == 0:
            return []

        boxes, scores, cls_ids = self._per_view_pipeline(boxes, scores, cls_ids)
        return self._build_results(boxes, scores, cls_ids)

    def _postprocess(
        self,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        orig_size: tuple[int, int],
    ) -> list[BoundingBox]:
        if output.ndim == 2 and output.shape[1] >= 6:
            return self._decode_final_dets(output, ratio, pad, orig_size)
        if output.ndim == 3 and output.shape[0] == 1 and output.shape[2] == 6:
            return self._decode_final_dets(output, ratio, pad, orig_size)
        return self._decode_raw_yolo(output, ratio, pad, orig_size)

    def _predict_single(self, image: np.ndarray) -> list[BoundingBox]:
        if image is None:
            raise ValueError("Input image is None")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Input is not numpy array: {type(image)}")
        if image.ndim != 3:
            raise ValueError(f"Expected HWC image, got shape={image.shape}")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Invalid image shape={image.shape}")
        if image.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got shape={image.shape}")
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        input_tensor, ratio, pad, orig_size = self._preprocess(image)
        expected = (1, 3, self.input_height, self.input_width)
        if input_tensor.shape != expected:
            raise ValueError(
                f"Bad input tensor shape={input_tensor.shape}, expected={expected}"
            )

        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return self._postprocess(outputs[0], ratio, pad, orig_size)

    def _predict_tta(self, image: np.ndarray) -> list[BoundingBox]:
        """Horizontal-flip TTA.
        Strategy:
          1. Predict on original and on flipped image.
          2. Map flipped boxes back to original coordinates.
          3. Per-class hard NMS on the union.
          4. For each kept box, compute the max same-class score across the
             FULL union (not just the post-NMS subset) -- this lets a high-
             confidence flipped detection raise a borderline original one.
          5. Cross-class dedup to suppress same-physical-object multi-class.
        """
        boxes_orig = self._predict_single(image)
        flipped = cv2.flip(image, 1)
        boxes_flip = self._predict_single(flipped)
        w = image.shape[1]
        boxes_flip = [
            BoundingBox(
                x1=w - b.x2, y1=b.y1, x2=w - b.x1, y2=b.y2,
                cls_id=b.cls_id, conf=b.conf,
            )
            for b in boxes_flip
        ]
        all_boxes = boxes_orig + boxes_flip
        if not all_boxes:
            return []

        coords = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in all_boxes], dtype=np.float32
        )
        scores = np.array([b.conf for b in all_boxes], dtype=np.float32)
        cls_ids = np.array([b.cls_id for b in all_boxes], dtype=np.int32)

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
                x1=int(round(kept_coords[j, 0])),
                y1=int(round(kept_coords[j, 1])),
                x2=int(round(kept_coords[j, 2])),
                y2=int(round(kept_coords[j, 3])),
                cls_id=int(kept_cls[j]),
                conf=float(boosted[j]),
            )
            for j in range(len(kept_coords))
        ]

    def _predict_tiles(self, image: np.ndarray) -> list[BoundingBox]:
        """Tile-based TTA for high-resolution images.
        Splits the source image into two overlapping horizontal tiles, runs
        single-pass inference on each at native scale, and translates boxes
        back to the global frame. Useful when source width >> model input
        width because letterboxing otherwise discards effective resolution
        that small / distant signs depend on.
        Returns an empty list if the image isn't wide enough to benefit; the
        caller falls back to the regular pipeline in that case.
        """
        h, w = image.shape[:2]
        if w < int(self.input_width * self.tile_trigger_ratio):
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
        """Merge boxes from multiple views (single / hflip / tiles).
        Same logic as `_predict_tta`'s tail: per-class hard NMS to dedupe,
        then for each kept box take the max same-class score across the full
        candidate union — a high-confidence detection in any view boosts
        borderline matches in others.
        """
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
                x1=int(round(kept_coords[j, 0])),
                y1=int(round(kept_coords[j, 1])),
                x2=int(round(kept_coords[j, 2])),
                y2=int(round(kept_coords[j, 3])),
                cls_id=int(kept_cls[j]),
                conf=float(boosted[j]),
            )
            for j in range(len(kept_coords))
        ]

    def _predict_full(self, image: np.ndarray) -> list[BoundingBox]:
        """Top-level per-frame prediction with all enabled augmentations.
        - `use_tta=True`: original + horizontal flip
        - `use_tile_tta=True` AND image wide enough: two overlapping tiles
        All views are merged via per-class NMS + cluster-max score boost.
        """
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
        for frame_number_in_batch, image in enumerate(batch_images):
            try:
                boxes = self._predict_full(image)
            except Exception as e:
                print(
                    f"⚠️ Inference failed for frame "
                    f"{offset + frame_number_in_batch}: {e}"
                )
                boxes = []
            results.append(
                TVFrameResult(
                    frame_id=offset + frame_number_in_batch,
                    boxes=boxes,
                    keypoints=[(0, 0) for _ in range(max(0, int(n_keypoints)))],
                )
            )
        return results
