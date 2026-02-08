from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from .phase_features import (
    compute_hand_object_distance, compute_finger_spread, compute_velocity,
    compute_fingertip_convergence, compute_hand_velocity
)
from ..extraction.trajectory import HandTrajectory, ObjectTrajectory

@dataclass
class PhaseSegment:
    label: str
    start_frame: int
    end_frame: int
    spatial_params: dict = None

class PhaseDetector:
    """
    Detects manipulation phases using heuristic thresholds and temporal consistency.
    """
    def __init__(self, fps: float = 30.0, consistency_frames: int = 3):
        self.fps = fps
        self.consistency_frames = consistency_frames
        self.release_consistency_frames = 5  # Frames needed for release detection

        # Thresholds (normalized units) - calibrated from real data
        self.approach_dist_threshold = 0.25
        self.transport_velocity_threshold = 0.02
        self.grasp_convergence_threshold = 0.23  # Fingertips closed (min seen: 0.17-0.20)
        self.release_convergence_threshold = 0.30  # Fingertips open (seen: 0.32-0.45)

    def detect_phases(self, hand_traj: HandTrajectory, obj_traj: ObjectTrajectory) -> List[PhaseSegment]:
        # Use smoothed features
        distances = compute_hand_object_distance(hand_traj, obj_traj, smooth=True)
        fingertip_conv = compute_fingertip_convergence(hand_traj, smooth=True)
        hand_speed = compute_hand_velocity(hand_traj, self.fps, smooth=True)
        obj_velocities = compute_velocity(obj_traj.centroids, self.fps)
        obj_speed = np.linalg.norm(obj_velocities, axis=1)

        # Compute convergence velocity (negative = closing)
        conv_vel = compute_velocity(fingertip_conv.reshape(-1, 1), self.fps).flatten()

        num_frames = len(distances)
        phases = []

        current_phase = "IDLE"
        start_idx = 0

        # Track minimum convergence seen during approach (for grasp detection)
        min_convergence = float('inf')
        convergence_stable_count = 0

        # For temporal consistency
        potential_phase = current_phase
        potential_count = 0

        for i in range(num_frames):
            dist = distances[i]
            convergence = fingertip_conv[i]
            h_speed = hand_speed[i]
            o_speed = obj_speed[i]
            c_vel = conv_vel[i]

            target_phase = current_phase

            # State machine logic
            if current_phase == "IDLE":
                if dist < 0.4:
                    target_phase = "APPROACH"
                    min_convergence = convergence

            elif current_phase == "APPROACH":
                # Track minimum convergence (fingertips getting closer)
                if convergence < min_convergence:
                    min_convergence = convergence
                    convergence_stable_count = 0
                else:
                    convergence_stable_count += 1

                # GRASP detection:
                # 1. Fingertips have converged below threshold
                # 2. Convergence is stable (not changing much)
                # 3. Hand has slowed down
                fingertips_closed = convergence < self.grasp_convergence_threshold
                convergence_stable = abs(c_vel) < 0.5 and convergence_stable_count >= 2
                hand_stable = h_speed < 0.8

                if fingertips_closed and convergence_stable and hand_stable:
                    target_phase = "GRASP"
                # Also transition if convergence drops significantly
                elif convergence < self.grasp_convergence_threshold:
                    target_phase = "GRASP"

            elif current_phase == "GRASP":
                # If fingertips clearly open -> RELEASE
                if convergence > self.release_convergence_threshold:
                    target_phase = "RELEASE"
                # If object is moving, it's TRANSPORT
                elif o_speed > self.transport_velocity_threshold * 2.0:
                    target_phase = "TRANSPORT"

            elif current_phase == "TRANSPORT":
                # If fingertips clearly spread -> RELEASE
                if convergence > self.release_convergence_threshold:
                    target_phase = "RELEASE"

            elif current_phase == "RELEASE":
                # If hand moves away from object -> RETREAT
                if dist > 0.18:
                    target_phase = "RETREAT"
                # If fingertips close again -> back to GRASP
                elif convergence < self.grasp_convergence_threshold:
                    target_phase = "GRASP"

            # Apply temporal consistency
            if target_phase != current_phase:
                if target_phase == potential_phase:
                    potential_count += 1
                else:
                    potential_phase = target_phase
                    potential_count = 1

                # Require more frames for RELEASE to avoid false triggers
                required_frames = self.release_consistency_frames if target_phase == "RELEASE" else self.consistency_frames

                if potential_count >= required_frames:
                    # Commit to new phase
                    phases.append(PhaseSegment(current_phase, start_idx, i))
                    current_phase = target_phase
                    start_idx = i
                    potential_count = 0
                    # Reset tracking for new phase
                    if target_phase == "APPROACH":
                        min_convergence = convergence
                        convergence_stable_count = 0
            else:
                potential_count = 0

        # Append the final phase
        if start_idx < num_frames:
            phases.append(PhaseSegment(current_phase, start_idx, num_frames - 1))

        return phases
