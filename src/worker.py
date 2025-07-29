from PySide6.QtCore import Qt, QTimer, Signal, QThread
import cv2
import numpy as np
from src.processors import compute_flow, compute_flowtrace
from src.utils import visualize_flow, visualize_flowtrace

class VideoLoaderThread(QThread):
    metadata_ready = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            metadata = {
                'resolution': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                'codec': cap.get(cv2.CAP_PROP_FOURCC)
            }
            cap.release()
            self.metadata_ready.emit(metadata)
        except Exception as e:
            self.error.emit(str(e))

class FlowProcessorThread(QThread):
    progress = Signal(int)
    frame_ready = Signal(np.ndarray, np.ndarray)
    error = Signal(str)

    def __init__(self, video_path, params):
        super().__init__()
        self.video_path = video_path
        self.params = params

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            start = self.params['start_frame']
            end = self.params['end_frame']
            step = self.params['step']
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if self.params['algorithm'] == "FlowTrace":
                trace_length = self.params['win_size']
                if trace_length > (end - start + 1):
                    raise ValueError("Trace length exceeds selected frame range")
                frame_buffer = []
                processed = 0
                total_windows = max(0, ((end - start + 1) - trace_length + 1) // step)
                for i in range(start, end + 1, step):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame_buffer.append(gray)
                    if len(frame_buffer) > trace_length:
                        frame_buffer.pop(0)
                    if len(frame_buffer) == trace_length:
                        stack = np.stack(frame_buffer, axis=0)
                        trace = compute_flowtrace(stack, self.params)
                        trace_viz = visualize_flowtrace(trace, frame.shape, self.params)
                        self.frame_ready.emit(frame, trace_viz)
                        processed += 1
                        if total_windows > 0:
                            self.progress.emit(int((processed / total_windows) * 100))
                cap.release()
            else:
                prev_gray = None
                processed = 0
                total_pairs = max(0, ((end - start) // step))
                for i in range(start, end + 1, step):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        flow = compute_flow(prev_gray, gray, self.params)
                        flow_viz = visualize_flow(flow, frame.shape, self.params)
                        self.frame_ready.emit(frame, flow_viz)
                        processed += 1
                        if total_pairs > 0:
                            self.progress.emit(int((processed / total_pairs) * 100))
                    prev_gray = gray
                cap.release()
        except Exception as e:
            self.error.emit(str(e))