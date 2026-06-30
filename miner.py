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
    input_size = 1024
    num_queries = 300
    interp = cv2.INTER_AREA
    conf_thres = 0.42
    nms_iou = 0.5
    max_det = 300
    min_side = 8.0
    min_box_area = 9.0
    max_aspect_ratio = 12.0

    rescue_bonus = 0.10   # floor = conf_thres - rescue_bonus = 0.32

    use_cuda_graph = False

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

        active = self.session.get_providers()[0]
        print(f"Model loaded from {model_path}  provider={active}  iobinding=ON")

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
            # CUDA EP keeps outputs on device; CPU buffer_ptr binding is not filled.
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

    def _letterbox(self, image: ndarray) -> tuple[float, int, int]:
        """Letterbox into pre-allocated buffers; writes model input to _input_tensor."""
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
            # CUDA EP: bind_cpu_input after clear_binding_inputs() raises
            # "Invalid argument" on chutes; use session.run instead.
            _, boxes, scores = self.session.run(
                self.output_names,
                {self.input_name: self._input_tensor, self.size_name: self._sizes},
            )
            np.copyto(self._out_boxes, boxes)
            np.copyto(self._out_scores, scores)
            return

        self.session.run_with_iobinding(self._io_binding)

    def _decode(self, boxes, scores, ratio, pad_w, pad_h, orig_w, orig_h):
        primary = scores > self.conf_thres

        # Rescue: if nothing clears the threshold, admit the single best
        # candidate when it clears the rescue floor.  This fires at most once
        # per frame and adds zero extra inference calls.
        if not primary.any() and len(scores) > 0:
            best = int(np.argmax(scores))
            if scores[best] >= self.conf_thres - self.rescue_bonus:
                primary = np.zeros(len(scores), dtype=bool)
                primary[best] = True

        boxes = boxes[primary].astype(np.float32)
        scores = scores[primary]
        if boxes.shape[0] == 0:
            return []

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        # Sanity filter: drop degenerate/tiny/image-spanning/extreme-AR boxes.
        bw = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        bh = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        area = bw * bh
        ar = np.where(
            (bw > 0) & (bh > 0),
            np.maximum(bw / np.maximum(bh, 1e-6), bh / np.maximum(bw, 1e-6)),
            np.inf,
        )
        sane = (
            (bw >= self.min_side) & (bh >= self.min_side)
            & (area >= self.min_box_area)
            & (area <= 0.95 * float(orig_w * orig_h))
            & (ar <= self.max_aspect_ratio)
        )
        boxes, scores = boxes[sane], scores[sane]
        if boxes.shape[0] == 0:
            return []

        if boxes.shape[0] > 1:
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
