from pathlib import Path
import math

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
    """
    YOLOv26 ONNX miner for road sign detection.
    
    Classes: broom, drainage gate, nozzle, track
    
    v26 is NMS-free — output shape: [1, 300, 6] (x1, y1, x2, y2, conf, cls_id).
    
    Features:
      - Vectorized NMS + sanity filter + dedup + flip TTA
      - Per-class rescue bonus (saves hard-to-detect classes at slightly lower conf)
      - Confidence boost from same-class cluster (TTA consensus)
      - Aggressive same-class overlap suppression
      - Per-class IoU thresholds
    """

    class_names = ['road sign']
    input_size = 1536
    cross_iou_thresh = 0.8
    max_det = 300
    #overlap_suppress_threshold = 0.85
    
    # Per-class confidence thresholds
    _conf_thres_array = np.array([0.35], dtype=np.float32)
    
    # Per-class IoU thresholds for same-class NMS
    _iou_thres_array = np.array([0.7], dtype=np.float32)
    
    # Per-class rescue bonus
    _bonus_array = np.array([0.2], dtype=np.float32)
    
    # Per-class minimum box area
    # Indices: 0=broom, 1=drainage gate, 2=nozzle, 3=track
    _min_box_area_array = np.array([16.0], dtype=np.float32)
    

    def __init__(self, path_hf_repo: Path) -> None:
        self.path_hf_repo = path_hf_repo
        
        print("ORT version:", ort.__version__)
        
        try:
            ort.preload_dlls()
            print("preload_dlls success")
        except Exception as e:
            print(f"preload_dlls failed: {e}")
        
        print("ORT available providers BEFORE session:", ort.get_available_providers())
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            str(path_hf_repo / "weights.onnx"),
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("Created ORT session with preferred CUDA provider list")
        print("ORT session providers:", self.session.get_providers())
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        input_shape = self.session.get_inputs()[0].shape
        
        self.input_h = self._safe_dim(input_shape[2], default=self.input_size)
        self.input_w = self._safe_dim(input_shape[3], default=self.input_size)
        self._warmup()

    def _warmup(self, iters: int = 3) -> None:
        try:
            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            for _ in range(max(1, iters)):
                self.predict_batch(batch_images=[dummy], offset=0, n_keypoints=0)
            print(f"✅ warmup: {iters} dummy predict_batch call(s) done")
        except Exception as e:
            print(f"⚠️ warmup skipped: {e}")

    def __repr__(self) -> str:
        return f"YOLOv26 road sign Miner classes={len(self.class_names)}"

    @staticmethod
    def _safe_dim(value, default: int) -> int:
        return value if isinstance(value, int) and value > 0 else default

    # ─── Preprocessing ────────────────────────────────────────────
    
    def _letterbox(
        self, image: ndarray, new_shape: tuple[int, int],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[ndarray, float, float, float]:
        orig_h, orig_w = image.shape[:2]
        target_w, target_h = new_shape
        
        r = min(target_w / orig_w, target_h / orig_h)
        new_unpad_w = int(round(orig_w * r))
        new_unpad_h = int(round(orig_h * r))
        
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        
        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h
        pad_w = dw / 2.0
        pad_h = dh / 2.0
        
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color,
        )
        return out, r, pad_w, pad_h

    def _preprocess(self, image_bgr: ndarray) -> tuple[np.ndarray, dict]:
        orig_h, orig_w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img, ratio, pad_w, pad_h = self._letterbox(rgb, (self.input_w, self.input_h))
        x = img.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        x = np.ascontiguousarray(x)
        
        return x, {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "ratio": ratio,
            "pad_w": pad_w,
            "pad_h": pad_h,
        }

    # ─── Vectorized box operations ───────────────────────────────
    
    @staticmethod
    def _clip_boxes(boxes: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        w, h = image_size
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    @staticmethod
    def _hard_nms(boxes: np.ndarray, scores: np.ndarray,
                  iou_thresh: float) -> np.ndarray:
        """Vectorized NMS. Returns indices to keep."""
        n = len(boxes)
        if n == 0:
            return np.array([], dtype=np.intp)
        order = np.argsort(-scores)
        keep = []
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

    def _per_class_hard_nms(self, boxes: np.ndarray, scores: np.ndarray,
                            cls_ids: np.ndarray) -> np.ndarray:
        """Per-class NMS using per-class IoU thresholds."""
        if len(boxes) == 0:
            return np.array([], dtype=np.intp)
        all_keep = []
        for c in np.unique(cls_ids):
            mask = cls_ids == c
            indices = np.where(mask)[0]
            cls_iou = float(self._iou_thres_array[c])  # per-class IoU threshold
            keep = self._hard_nms(boxes[mask], scores[mask], cls_iou)
            all_keep.extend(indices[keep].tolist())
        all_keep.sort()
        return np.array(all_keep, dtype=np.intp)

    def _cross_class_dedup_op(self, boxes: np.ndarray, scores: np.ndarray,
                              cls_ids: np.ndarray, iou_thresh: float
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
        keep = []
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

    def _filter_sane_boxes(self, boxes: np.ndarray, scores: np.ndarray,
                       cls_ids: np.ndarray, orig_size: tuple[int, int]
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter by per-class min area, max area ratio, and aspect ratio."""
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        
        orig_w, orig_h = orig_size
        image_area = float(orig_w * orig_h)
        bw = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        bh = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        area = bw * bh
        
        ar = np.where(
            (bw > 0) & (bh > 0),
            np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6)),
            np.inf,
        )
        
        # Per-class minimum area
        class_min_area = self._min_box_area_array[cls_ids]
        
        keep = (
            (area >= class_min_area) &
            (area <= 0.95 * image_area)
        )
        return boxes[keep], scores[keep], cls_ids[keep]

    def _max_score_per_cluster(self, post_boxes: np.ndarray,
                               post_cls: np.ndarray,
                               full_boxes: np.ndarray,
                               full_scores: np.ndarray,
                               full_cls: np.ndarray,
                               iou_thresh: float) -> np.ndarray:
        """For each kept box, set confidence to max score in its SAME-CLASS cluster."""
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

    def _conf_filter_mask(self, scores: np.ndarray,
                          cls_ids: np.ndarray) -> np.ndarray:
        """Per-class threshold with rescue bonus for missed classes."""
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

    def _suppress_overlapping_same_class(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        cls_ids: np.ndarray,
        threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Drop a same-class box that is (almost) entirely *contained* inside a larger
        same-class box — a duplicate detection of one object.

        Containment is intersection / area_of_SMALLER_box (IoMin), NOT IoU.
        A small box nested in a large one has tiny IoU, so plain NMS never removes
        it; IoMin catches it.

            black (large) + green (fully inside black)  -> same gate, drop green
            red   (sticks out of black)                 -> separate gate, keep

        Survivor = the LARGER box, and its confidence is raised to the cluster max.
        """
        n = len(boxes)
        if n <= 1:
            return boxes, scores, cls_ids

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).copy()
        cls_ids = np.asarray(cls_ids, dtype=np.int32)

        areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) *
                np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))

        keep = np.ones(n, dtype=bool)

        # Largest first, so the survivor of a containment chain is the biggest box.
        order = np.argsort(-areas)

        for idx_a in range(n):
            a = order[idx_a]
            if not keep[a]:
                continue
            for idx_b in range(idx_a + 1, n):
                b = order[idx_b]              # areas[b] <= areas[a]
                if not keep[b]:
                    continue
                if cls_ids[a] != cls_ids[b]:
                    continue

                x1 = max(boxes[a, 0], boxes[b, 0])
                y1 = max(boxes[a, 1], boxes[b, 1])
                x2 = min(boxes[a, 2], boxes[b, 2])
                y2 = min(boxes[a, 3], boxes[b, 3])
                if x2 <= x1 or y2 <= y1:
                    continue
                inter = (x2 - x1) * (y2 - y1)

                # How much of the SMALLER box (b) lies inside the larger (a):
                containment_b = inter / max(areas[b], 1e-9)

                if containment_b >= threshold:        # b is nested -> it's a duplicate
                    scores[a] = max(scores[a], scores[b])   # keep the higher score
                    keep[b] = False                          # drop the smaller (green)

        keep_idx = np.where(keep)[0]
        return boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]

    def _per_view_pipeline(self, boxes: np.ndarray, scores: np.ndarray,
                           cls_ids: np.ndarray, orig_size: tuple[int, int]
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sanity filter + per-class NMS + cross-class dedup."""
        boxes, scores, cls_ids = self._filter_sane_boxes(
            boxes, scores, cls_ids, orig_size
        )
        if len(boxes) == 0:
            return boxes, scores, cls_ids
        if len(boxes) > 1:
            keep = self._per_class_hard_nms(boxes, scores, cls_ids)
            boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
        if len(scores) > self.max_det:
            top = np.argsort(-scores)[: self.max_det]
            boxes, scores, cls_ids = boxes[top], scores[top], cls_ids[top]
        if len(boxes) > 1:
            boxes, scores, cls_ids = self._cross_class_dedup_op(
                boxes, scores, cls_ids, self.cross_iou_thresh
            )
        return boxes, scores, cls_ids

    # ─── v26-specific decoding ────────────────────────────────────
    
    def _decode_v26_output(self, preds: np.ndarray, ratio: float,
                           pad: tuple[float, float],
                           orig_size: tuple[int, int]) -> list[BoundingBox]:
        """Decode YOLOv26 output (shape [1, 300, 6] or [300, 6])."""
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]
        
        if preds.ndim != 2 or preds.shape[1] < 6:
            print(f"Warning: Unexpected v26 output shape: {preds.shape}")
            return []

        boxes = preds[:, :4].astype(np.float32)
        scores = preds[:, 4].astype(np.float32)
        cls_ids = preds[:, 5].astype(np.int32)

        n_cls = len(self.class_names)
        valid = (cls_ids >= 0) & (cls_ids < n_cls)
        boxes = boxes[valid]
        scores = scores[valid]
        cls_ids = cls_ids[valid]
        
        if len(boxes) == 0:
            return []

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

        boxes, scores, cls_ids = self._per_view_pipeline(
            boxes, scores, cls_ids, orig_size
        )
        
        return self._build_results(boxes, scores, cls_ids, orig_size)

    @staticmethod
    def _build_results(boxes: np.ndarray, scores: np.ndarray,
                       cls_ids: np.ndarray,
                       orig_size: tuple[int, int]) -> list[BoundingBox]:
        results = []
        orig_w, orig_h = orig_size
        for box, conf, cls_id in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                BoundingBox(
                    x1=max(0, min(orig_w, int(math.floor(x1)))),
                    y1=max(0, min(orig_h, int(math.floor(y1)))),
                    x2=max(0, min(orig_w, int(math.ceil(x2)))),
                    y2=max(0, min(orig_h, int(math.ceil(y2)))),
                    cls_id=int(cls_id),
                    conf=float(max(0.0, min(1.0, conf))),
                )
            )
        return results

    # ─── Single-view inference ────────────────────────────────────
    
    def _predict_single(self, image_bgr: np.ndarray) -> list[BoundingBox]:
        if image_bgr is None or not isinstance(image_bgr, np.ndarray):
            raise ValueError("Invalid image input")
        if image_bgr.dtype != np.uint8:
            image_bgr = image_bgr.astype(np.uint8)
        
        inp, meta = self._preprocess(image_bgr)
        outputs = self.session.run(None, {self.input_name: inp})
        
        ratio = float(meta["ratio"])
        pad = (float(meta["pad_w"]), float(meta["pad_h"]))
        orig_size = (int(meta["orig_w"]), int(meta["orig_h"]))
        
        return self._decode_v26_output(outputs[0], ratio, pad, orig_size)

    # ─── TTA inference ────────────────────────────────────────────
    
    def _infer_single(self, image_bgr: ndarray) -> list[BoundingBox]:
        """TTA: original + horizontal flip, with confidence boosting from same-class cluster."""
        boxes_orig = self._predict_single(image_bgr)
        # flipped = cv2.flip(image_bgr, 1)
        # boxes_flip = self._predict_single(flipped)
        
        # # Un-flip x coordinates
        # w = image_bgr.shape[1]
        # boxes_flip = [
        #     BoundingBox(
        #         x1=w - b.x2, y1=b.y1, x2=w - b.x1, y2=b.y2,
        #         cls_id=b.cls_id, conf=b.conf,
        #     )
        #     for b in boxes_flip
        # ]
        
        all_boxes = boxes_orig# + boxes_flip
        if not all_boxes:
            return []

        # Convert to arrays
        coords = np.array(
            [[b.x1, b.y1, b.x2, b.y2] for b in all_boxes], dtype=np.float32
        )
        scores = np.array([b.conf for b in all_boxes], dtype=np.float32)
        cls_ids = np.array([b.cls_id for b in all_boxes], dtype=np.int32)

        # Per-class NMS (uses per-class IoU thresholds)
        hard_keep = self._per_class_hard_nms(coords, scores, cls_ids)
        if len(hard_keep) == 0:
            return []
        
        if len(hard_keep) > self.max_det:
            top = np.argsort(-scores[hard_keep])[: self.max_det]
            hard_keep = hard_keep[top]
        
        # For confidence boost, use average IoU threshold (or could use median)
        avg_iou = float(np.mean(self._iou_thres_array))
        boosted = self._max_score_per_cluster(
            coords[hard_keep], cls_ids[hard_keep],
            coords, scores, cls_ids, avg_iou,
        )

        kept_coords = coords[hard_keep]
        kept_cls = cls_ids[hard_keep]
        
        # Cross-class dedup
        if len(kept_coords) > 1:
            kept_coords, boosted, kept_cls = self._cross_class_dedup_op(
                kept_coords, boosted, kept_cls, self.cross_iou_thresh
            )

        # if len(kept_coords) > 1:
        #     kept_coords, boosted, kept_cls = self._suppress_overlapping_same_class(
        #         kept_coords, boosted, kept_cls, self.overlap_suppress_threshold
        #     )
           
        # Build BoundingBox results
        orig_h, orig_w = image_bgr.shape[:2]
        out_boxes = []
        for j in range(len(kept_coords)):
            x1, y1, x2, y2 = kept_coords[j].tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            out_boxes.append(
                BoundingBox(
                    x1=max(0, min(orig_w, int(math.floor(x1)))),
                    y1=max(0, min(orig_h, int(math.floor(y1)))),
                    x2=max(0, min(orig_w, int(math.ceil(x2)))),
                    y2=max(0, min(orig_h, int(math.ceil(y2)))),
                    cls_id=int(kept_cls[j]),
                    conf=float(max(0.0, min(1.0, boosted[j]))),
                )
            )
        
        return out_boxes

    # ─── Public API ───────────────────────────────────────────────
    
    def predict_batch(
        self,
        batch_images: list[ndarray],
        offset: int,
        n_keypoints: int,
    ) -> list[TVFrameResult]:
        results: list[TVFrameResult] = []
        for idx, image in enumerate(batch_images):
            try:
                boxes = self._infer_single(image)
            except Exception as e:
                print(f"Inference failed for frame {offset + idx}: {e}")
                boxes = []
            keypoints = [(0, 0) for _ in range(max(0, int(n_keypoints)))]
            results.append(
                TVFrameResult(
                    frame_id=offset + idx,
                    boxes=boxes,
                    keypoints=keypoints,
                )
            )
        return results