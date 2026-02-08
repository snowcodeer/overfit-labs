import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from typing import Optional, List, Tuple
from .trajectory import HandTrajectory

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class MediaPipeTracker:
    """
    Wrapper for MediaPipe Tasks HandLandmarker to extract 21 hand landmarks per frame.
    Uses the modern Tasks API instead of the legacy solutions API.
    """
    def __init__(self, model_path: str = "hand_landmarker.task", max_num_hands: int = 1, min_detection_confidence: float = 0.5):
        self.model_path = model_path
        
        # Automatically download model if missing
        if not os.path.exists(self.model_path):
            print(f"Downloading MediaPipe hand landmarker model to {self.model_path}...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Process a single BGR frame and return landmarks and confidence.
        Note: In VIDEO mode, we need timestamps.
        """
        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Use a dummy timestamp or increment if not provided
        self.timestamp_ms += 33 # Assume ~30fps if called frame-by-frame
        
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        
        if result.hand_landmarks:
            # Take the first hand
            hand_landmarks = result.hand_landmarks[0]
            confidence = result.handedness[0][0].score
            
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            return landmarks, confidence
        
        return None, 0.0

    def process_video(self, video_path: str) -> HandTrajectory:
        """
        Processes an entire video and returns a HandTrajectory.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0
        
        landmarks_list = []
        confidences_list = []
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use real timestamps from video
            timestamp_ms = int(1000 * frame_idx / fps)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if result.hand_landmarks:
                lms = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
                conf = result.handedness[0][0].score
                landmarks_list.append(lms)
                confidences_list.append(conf)
            else:
                landmarks_list.append(np.zeros((21, 3)))
                confidences_list.append(0.0)
            
            frame_idx += 1
        
        cap.release()
        return HandTrajectory(
            landmarks=np.array(landmarks_list),
            confidences=np.array(confidences_list)
        )

    def process_frames(self, frames: List[np.ndarray], fps: float = 30.0) -> HandTrajectory:
        """
        Processes a list of BGR frames and returns a HandTrajectory.
        """
        landmarks_list = []
        confidences_list = []

        for frame_idx, frame in enumerate(frames):
            timestamp_ms = int(1000 * frame_idx / fps)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                lms = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]])
                conf = result.handedness[0][0].score
                landmarks_list.append(lms)
                confidences_list.append(conf)
            else:
                landmarks_list.append(np.zeros((21, 3)))
                confidences_list.append(0.0)

        return HandTrajectory(
            landmarks=np.array(landmarks_list),
            confidences=np.array(confidences_list)
        )

    def close(self):
        self.landmarker.close()
