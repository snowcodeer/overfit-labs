import numpy as np
from typing import List, Optional
from ..extraction.trajectory import HandTrajectory, ObjectTrajectory

def compute_hand_object_distance(hand_traj: HandTrajectory, obj_traj: ObjectTrajectory, smooth: bool = True) -> np.ndarray:
    """
    Compute Euclidean distance between palm center (avg of landmarks 5, 17) and object centroid.
    """
    # Use average of index MCP (5) and pinky MCP (17) as palm center
    palm_center = (hand_traj.landmarks[:, 5, :2] + hand_traj.landmarks[:, 17, :2]) / 2.0
    distances = np.linalg.norm(palm_center - obj_traj.centroids, axis=1)
    
    if smooth:
        distances = exponential_smoothing(distances, alpha=0.3)
    return distances

def compute_finger_spread(hand_traj: HandTrajectory, smooth: bool = True) -> np.ndarray:
    """
    Compute distance between thumb tip (4) and index tip (8).
    """
    thumb_tip = hand_traj.landmarks[:, 4, :3]
    index_tip = hand_traj.landmarks[:, 8, :3]
    spread = np.linalg.norm(thumb_tip - index_tip, axis=1)

    if smooth:
        spread = exponential_smoothing(spread, alpha=0.3)
    return spread


def compute_fingertip_convergence(hand_traj: HandTrajectory, smooth: bool = True) -> np.ndarray:
    """
    Compute average pairwise distance between all fingertips.
    Lower value = fingertips are close together (grasping).
    Fingertips: thumb(4), index(8), middle(12), ring(16), pinky(20)
    """
    fingertip_ids = [4, 8, 12, 16, 20]
    num_frames = hand_traj.landmarks.shape[0]
    avg_distances = np.zeros(num_frames)

    for i in range(num_frames):
        tips = hand_traj.landmarks[i, fingertip_ids, :3]
        # Compute pairwise distances
        total_dist = 0
        count = 0
        for j in range(len(fingertip_ids)):
            for k in range(j + 1, len(fingertip_ids)):
                total_dist += np.linalg.norm(tips[j] - tips[k])
                count += 1
        avg_distances[i] = total_dist / count if count > 0 else 0

    if smooth:
        avg_distances = exponential_smoothing(avg_distances, alpha=0.3)
    return avg_distances


def compute_hand_velocity(hand_traj: HandTrajectory, fps: float, smooth: bool = True) -> np.ndarray:
    """
    Compute hand velocity using wrist (0) movement.
    """
    wrist_pos = hand_traj.landmarks[:, 0, :3]
    velocity = compute_velocity(wrist_pos, fps)
    speed = np.linalg.norm(velocity, axis=1)

    if smooth:
        speed = exponential_smoothing(speed, alpha=0.3)
    return speed

def compute_palm_orientation(hand_traj: HandTrajectory) -> np.ndarray:
    """
    Compute palm normal vector using wrist (0), index MCP (5), and pinky MCP (17).
    Returns: (num_frames, 3) unit normal vectors.
    """
    wrist = hand_traj.landmarks[:, 0, :3]
    index_mcp = hand_traj.landmarks[:, 5, :3]
    pinky_mcp = hand_traj.landmarks[:, 17, :3]
    
    v1 = index_mcp - wrist
    v2 = pinky_mcp - wrist
    
    normals = np.cross(v1, v2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / (norms + 1e-6)

def exponential_smoothing(data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply simple exponential smoothing."""
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    return smoothed

def compute_velocity(trajectory: np.ndarray, fps: float) -> np.ndarray:
    """Compute velocity via finite difference."""
    dt = 1.0 / fps
    velocity = np.diff(trajectory, axis=0) / dt
    # Pad with zero to maintain length
    velocity = np.vstack([np.zeros((1, velocity.shape[1])), velocity])
    return velocity
