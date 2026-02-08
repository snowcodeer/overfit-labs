import numpy as np
from typing import Dict

def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class AdroitRetargeter:
    """
    Retargets MediaPipe hand landmarks to Adroit/ShadowHand joint angles.
    """
    def __init__(self):
        # Joint limits for Adroit (representative values, should be verified with MJCF)
        self.joint_limits = {
            "FFJ3": (-0.349, 0.349), # Index MCP abduction
            "FFJ2": (0, 1.57),      # Index MCP flexion
            "FFJ1": (0, 1.57),      # Index PIP flexion
            "FFJ0": (0, 1.57),      # Index DIP flexion
            # ... and so on for other fingers
        }

    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute Adroit-compatible joint angles from 21 MediaPipe landmarks.
        landmarks: (21, 3)
        """
        # MediaPipe Indices
        WRIST = 0
        THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
        RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
        PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
        
        angles = {}
        
        def get_finger_angles(mcp, pip, dip, tip, prefix):
            # PIP Flexion (J1)
            v_mcp_pip = landmarks[pip] - landmarks[mcp]
            v_pip_dip = landmarks[dip] - landmarks[pip]
            angles[f"{prefix}J1"] = angle_between(v_mcp_pip, v_pip_dip)
            
            # DIP Flexion (J0)
            v_dip_tip = landmarks[tip] - landmarks[dip]
            angles[f"{prefix}J0"] = angle_between(v_pip_dip, v_dip_tip)
            
            # MCP Flexion (J2) - Approx against palm plane normal or just wrist vector
            v_wrist_mcp = landmarks[mcp] - landmarks[WRIST]
            angles[f"{prefix}J2"] = angle_between(v_wrist_mcp, v_mcp_pip)
            
            # Abduction (J3) - difficult from single view, assume 0 or small
            angles[f"{prefix}J3"] = 0.0

        # Fingers
        get_finger_angles(INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, "FF")
        get_finger_angles(MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, "MF")
        get_finger_angles(RING_MCP, RING_PIP, RING_DIP, RING_TIP, "RF")
        get_finger_angles(PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, "LF")
        
        # Thumb (Adroit has 5 joints: THJ0-THJ4)
        # Simplify: Map IP to J0, MCP to J1, CMC to J2
        v_cmc_mcp = landmarks[THUMB_MCP] - landmarks[THUMB_CMC]
        v_mcp_ip = landmarks[THUMB_IP] - landmarks[THUMB_MCP]
        v_ip_tip = landmarks[THUMB_TIP] - landmarks[THUMB_IP]
        
        angles["THJ0"] = angle_between(v_mcp_ip, v_ip_tip) # IP Flexion
        angles["THJ1"] = angle_between(v_cmc_mcp, v_mcp_ip) # MCP Flexion
        angles["THJ2"] = angle_between(landmarks[THUMB_CMC] - landmarks[WRIST], v_cmc_mcp) # CMC Flexion
        angles["THJ3"] = 0.0 # Abduction
        angles["THJ4"] = 0.0 # Opposition
        
        return angles

    def retarget_sequence(self, hand_traj: np.ndarray) -> np.ndarray:
        """
        Retarget a sequence of landmarks to joint angle vectors.
        hand_traj: (num_frames, 21, 3)
        Returns: (num_frames, 28) including arm
        """
        num_frames = hand_traj.shape[0]
        joint_vectors = np.zeros((num_frames, 28))
        
        for i in range(num_frames):
            angles = self.compute_angles(hand_traj[i])
            # Map angles dict to fixed vector indices for Adroit
            # Placeholder mapping
    def retarget_sequence(self, hand_traj: np.ndarray) -> np.ndarray:
        """
        Retarget a sequence of landmarks to joint angle vectors.
        hand_traj: (num_frames, 21, 3)
        Returns: (num_frames, 30) suitable for AdroitHandRelocate-v1
        """
        num_frames = hand_traj.shape[0]
        joint_vectors = np.zeros((num_frames, 30))
        
        # Scaling factors for position (approximate)
        # MediaPipe is normalized [0,1]. Adroit workspace is roughly [-0.5, 0.5]?
        # We need to calibrate this. For now, use a reasonable scale.
        POS_SCALE = 1.0 
        POS_OFFSET = np.array([-0.2, -0.2, 0.5]) # Initial guess to put hand in workspace
        
        for i in range(num_frames):
            lms = hand_traj[i]
            angles = self.compute_angles(lms)
            
            # 1. Base Position (ARTx, ARTy, ARTz) - Indices 0-2
            # Use Wrist landmark (0)
            # Map MP (x, y, z) -> Adroit. 
            # MP: y down. Adroit: z up?
            # Let's try direct mapping 
            wrist_pos = lms[0]
            # Flip Y for Adroit (y is usually forward/side, z is up)
            # This needs tuning. I'll pass raw MP for now and scale.
            target_pos = (wrist_pos - 0.5) * 2.0 # Center to 0
            joint_vectors[i, 0] = target_pos[0] * POS_SCALE
            joint_vectors[i, 1] = target_pos[2] * POS_SCALE # usage depth as y?
            joint_vectors[i, 2] = -target_pos[1] * POS_SCALE + 0.5 # usage y as z?
            
            # 2. Base Rotation (ARRx, ARRy, ARRz) - Indices 3-5
            # TODO: Compute from Wrist-Index-Pinky plane
            joint_vectors[i, 3:6] = 0.0 # Keep fixed orientation for now
            
            # 3. Wrist (WRJ1, WRJ0) - Indices 6-7
            joint_vectors[i, 6] = 0.0 # Flexion (TODO)
            joint_vectors[i, 7] = 0.0 # Deviation (TODO)
            
            # 4. Fingers
            # FF (Index) - Indices 8-11: J3, J2, J1, J0
            joint_vectors[i, 8]  = angles.get("FFJ3", 0)
            joint_vectors[i, 9]  = angles.get("FFJ2", 0)
            joint_vectors[i, 10] = angles.get("FFJ1", 0)
            joint_vectors[i, 11] = angles.get("FFJ0", 0)
            
            # MF (Middle) - Indices 12-15: J3, J2, J1, J0
            joint_vectors[i, 12] = angles.get("MFJ3", 0)
            joint_vectors[i, 13] = angles.get("MFJ2", 0)
            joint_vectors[i, 14] = angles.get("MFJ1", 0)
            joint_vectors[i, 15] = angles.get("MFJ0", 0)
            
            # RF (Ring) - Indices 16-19: J3, J2, J1, J0
            joint_vectors[i, 16] = angles.get("RFJ3", 0)
            joint_vectors[i, 17] = angles.get("RFJ2", 0)
            joint_vectors[i, 18] = angles.get("RFJ1", 0)
            joint_vectors[i, 19] = angles.get("RFJ0", 0)
            
            # LF (Pinky) - Indices 20-24: J4, J3, J2, J1, J0
            # Note: compute_angles only gave J3..J0. J4 is palm arch.
            joint_vectors[i, 20] = 0.0 # LFJ4
            joint_vectors[i, 21] = angles.get("LFJ3", 0)
            joint_vectors[i, 22] = angles.get("LFJ2", 0)
            joint_vectors[i, 23] = angles.get("LFJ1", 0)
            joint_vectors[i, 24] = angles.get("LFJ0", 0)
            
            # TH (Thumb) - Indices 25-29: J4, J3, J2, J1, J0
            joint_vectors[i, 25] = angles.get("THJ4", 0)
            joint_vectors[i, 26] = angles.get("THJ3", 0)
            joint_vectors[i, 27] = angles.get("THJ2", 0)
            joint_vectors[i, 28] = angles.get("THJ1", 0)
            joint_vectors[i, 29] = angles.get("THJ0", 0)
            
        return joint_vectors
