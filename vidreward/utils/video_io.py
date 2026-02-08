import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import os

class VideoReader:
    """
    Wrapper for reading video frames efficiently.
    Uses OpenCV for now, but can be extended to use decord for faster seek.
    """
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields BGR frames."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class VideoWriter:
    """
    Wrapper for writing processed frames to a video file.
    Ensures frames are resized to target dimensions to prevent corruption.
    """
    def __init__(self, output_path: str, fps: float, width: int, height: int):
        self.width = width
        self.height = height
        # Try mp4v first as it's more stable on some Windows setups
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Fallback to avc1 if mp4v is not available
        if not self.writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Final fallback to XVID
        if not self.writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write_frame(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
