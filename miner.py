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
    interp = cv2.INTER_AREA
    conf_thres = 0.34
    nms_iou = 0.6
    max_det = 300

    def __init__(self, path_hf_repo: Path) -> None:
        model_path = Path(path_hf_repo) / "weights.onnx"

        try:
            ort.preload_dlls()
        except Exception:
            pass

        providers = self._provider_list_with_opts()
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                disabled_optimizers=["SimplifiedLayerNormFusion"],
                providers=providers,
            )
        except Exception:
            try:
                self.session = ort.InferenceSession(str(model_path), providers=providers)
            except Exception:
                self.session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )

        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"Model loaded from {model_path}")

    @staticmethod
    def _providers() -> list[str]:
        avail = ort.get_available_providers()
        return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]

    def _provider_list_with_opts(self):
        out = []
        for p in self._providers():
            if p == "CUDAExecutionProvider":
                out.append((p, {"cudnn_conv_algo_search": "HEURISTIC"}))
            else:
                out.append(p)
        return out

    @staticmethod
    def _nms(boxes: ndarray, scores: ndarray, iou_thres: float) -> list[int]:
        if boxes.shape[0] == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0.0)
            order = rest[iou <= iou_thres]
        return keep

    def _letterbox(self, image: ndarray):
        size = self.input_size
        oh, ow = image.shape[:2]
        ratio = min(size / ow, size / oh)
        nw, nh = int(ow * ratio), int(oh * ratio)

        resized = cv2.resize(image, (nw, nh), interpolation=self.interp)
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        pad_w, pad_h = (size - nw) // 2, (size - nh) // 2
        canvas[pad_h : pad_h + nh, pad_w : pad_w + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return np.ascontiguousarray(chw), ratio, pad_w, pad_h

    def _decode(self, boxes, scores, ratio, pad_w, pad_h, orig_w, orig_h):
        keep = scores > self.conf_thres
        boxes = boxes[keep]
        scores = scores[keep]
        if boxes.shape[0] == 0:
            return []

        boxes = boxes.astype(np.float32, copy=True)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        if self.nms_iou is not None and boxes.shape[0] > 1:
            kept = self._nms(boxes, scores, self.nms_iou)
            boxes, scores = boxes[kept], scores[kept]

        if scores.shape[0] > self.max_det:
            top = np.argsort(-scores)[: self.max_det]
            boxes, scores = boxes[top], scores[top]

        out: list[BoundingBox] = []
        for (x1, y1, x2, y2), conf in zip(boxes, scores):
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(
                BoundingBox(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    cls_id=0, conf=float(conf),
                )
            )
        return out

    def _predict_single(self, image: ndarray) -> list[BoundingBox]:
        chw, ratio, pad_w, pad_h = self._letterbox(image)
        tensor = chw[None, ...]
        sizes = np.array([[self.input_size, self.input_size]], dtype=np.int64)
        _, boxes, scores = self.session.run(
            self.output_names, {self.input_name: tensor, self.size_name: sizes}
        )
        return self._decode(
            boxes[0], scores[0], ratio, pad_w, pad_h, image.shape[1], image.shape[0]
        )

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
                frame_boxes = self._predict_single(image)
            except Exception:
                frame_boxes = []

            results.append(TVFrameResult(
                frame_id=frame_id,
                boxes=frame_boxes,
                keypoints=list(kp),
            ))

        return results
