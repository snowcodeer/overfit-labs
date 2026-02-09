import gymnasium as gym
import gymnasium_robotics
import numpy as np
import cv2
import argparse
import os
import mujoco

from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
from vidreward.extraction.vision import detect_rubiks_cube_classical
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.extraction.trajectory import ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector


def get_finger_contacts(model, data, obj_geom_id):
    """
    Check which fingers are in contact with the object.
    Returns dict: {finger_name: is_in_contact}
    Detects contact with any finger segment (proximal, middle, distal).
    """
    # All finger geoms in Adroit (not just tips)
    finger_geoms = {
        'ff': ['C_ffproximal', 'C_ffmiddle', 'C_ffdistal'],   # Index
        'mf': ['C_mfproximal', 'C_mfmiddle', 'C_mfdistal'],   # Middle
        'rf': ['C_rfproximal', 'C_rfmiddle', 'C_rfdistal'],   # Ring
        'lf': ['C_lfproximal', 'C_lfmiddle', 'C_lfdistal'],   # Pinky
        'th': ['C_thproximal', 'C_thmiddle', 'C_thdistal'],   # Thumb
    }

    contacts = {name: False for name in finger_geoms}

    # Build geom ID -> finger name mapping
    geom_to_finger = {}
    for finger, geom_names in finger_geoms.items():
        for geom_name in geom_names:
            try:
                gid = model.geom(geom_name).id
                geom_to_finger[gid] = finger
            except:
                pass  # Geom not found

    # Check all contacts
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        # Check if contact involves object
        if geom1 == obj_geom_id or geom2 == obj_geom_id:
            other_geom = geom2 if geom1 == obj_geom_id else geom1
            # Check if other geom is any finger segment
            if other_geom in geom_to_finger:
                contacts[geom_to_finger[other_geom]] = True

    return contacts


def replay_m1(video_path: str, output_path: str = "adroit_replay.mp4", 
              obj_scale: float = 0.8, 
              hand_scales: tuple = (0.5, 0.3, 0.3), 
              grasp_offset: tuple = (0.0, -0.08, 0.08),
              analysis_path: str = None):
    print(f"Processing video: {video_path}")
    print(f"Config: ObjScale={obj_scale}, HandScales={hand_scales}, GraspOffset={grasp_offset}")

    # Mirror video for left hand -> right hand conversion
    # Set to False for right-handed videos
    FLIP_VIDEO = False

    # 1. Load video frames first
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())

    if FLIP_VIDEO:
        frames = [cv2.flip(f, 1) for f in frames]  # 1 = horizontal flip
        print("Video mirrored horizontally (left hand -> right hand)")

    # 2. Extract Hand Trajectory from (possibly mirrored) frames
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)  # Track on mirrored frames
    tracker.close()

    h_vid, w_vid = frames[0].shape[:2]
    num_frames = len(frames)
    
    obj_pos_sim = np.zeros((num_frames, 3))
    obj_pos_vid_norm = np.zeros((num_frames, 2))  # Object position in normalized video space [0,1]
    
    # Basic tracking
    bbox = None
    for i in range(min(50, num_frames)):
        bbox = detect_rubiks_cube_classical(frames[i])
        if bbox: break
        
    if bbox is None:
        print("Warning: Object not detected clearly. Using center.")
        curr_box = (w_vid//2 - 40, h_vid//2 - 40, 80, 80)
    else:
        curr_box = bbox

    # Sim Workspace approx: X[-0.3, 0.3], Y[-0.3, 0.3], Z[0.035] (table)
    # Video Coords: X[0, w], Y[0, h]
    # Simple mapping: Center video -> Center table
    
    sim_tracker = cv2.TrackerCSRT_create()
    sim_tracker.init(frames[0], curr_box)
    
    for i in range(num_frames):
        success, box = sim_tracker.update(frames[i])
        if success:
            curr_box = box
        else:
             # Try re-detect
             det = detect_rubiks_cube_classical(frames[i])
             if det:
                 curr_box = det
                 sim_tracker = cv2.TrackerCSRT_create()
                 sim_tracker.init(frames[i], curr_box)
        
        bx, by, bw, bh = curr_box
        cx, cy = bx + bw/2, by + bh/2

        # Store normalized video position [0, 1] for delta computation
        obj_pos_vid_norm[i] = [cx / w_vid, cy / h_vid]

        # Normalize to [-0.5, 0.5] for sim mapping
        nx = (cx / w_vid) - 0.5
        ny = (cy / h_vid) - 0.5

        # Map to Sim coordinates
        SCALE = obj_scale
        obj_pos_sim[i] = [nx * SCALE, -ny * SCALE + 0.1, 0.035]  # Z fixed on table

    # Detect GRASP phase to calibrate object position
    if analysis_path and os.path.exists(analysis_path):
        import json
        with open(analysis_path, 'r') as f:
            analysis_data = json.load(f)
            
        # Extract milestones or fallback to legacy fields
        milestones = analysis_data.get("milestones", [])
        if milestones:
            # Mapping heuristic for simulation physics:
            # Identify grasp and release frames from the label names
            grasp_ms = next((m for m in milestones if "grasp" in m['label'].lower() or "hold" in m['label'].lower() or "contact" in m['label'].lower()), milestones[0])
            release_ms = next((m for m in milestones if "release" in m['label'].lower() or "throw" in m['label'].lower() or "drop" in m['label'].lower()), milestones[-1])
            
            grasp_frame = grasp_ms['frame']
            release_frame = release_ms['frame']
        else:
            grasp_frame = analysis_data.get("grasp_frame", 0)
            release_frame = analysis_data.get("release_frame", num_frames)
            
        print(f"Using milestones from analysis: {grasp_frame}, {release_frame}")
        
        # Build manual phases for logic compatibility
        from dataclasses import dataclass
        @dataclass
        class SimplePhase:
            label: str
            start_frame: int
            end_frame: int
            
        phases = [
            SimplePhase("APPROACH", 0, grasp_frame - 1),
            SimplePhase("GRASP", grasp_frame, grasp_frame + 10), # Small window for initial grasp physics
            SimplePhase("TRANSPORT", grasp_frame + 1, release_frame - 1),
            SimplePhase("RELEASE", release_frame, num_frames)
        ]
    else:
        print("Detecting grasp phase...")
        obj_traj = ObjectTrajectory(centroids=obj_pos_vid_norm)
        phase_detector = PhaseDetector()
        phases = phase_detector.detect_phases(hand_traj, obj_traj)

        # Find first GRASP frame
        grasp_frame = 0
        for phase in phases:
            if phase.label == "GRASP":
                grasp_frame = phase.start_frame
                break

    print(f"First grasp at frame: {grasp_frame}")

    # Use MIDDLE FINGER MCP (landmark 9) for grasp positioning - closer to actual contact point
    GRASP_LANDMARK = 9  # Middle finger MCP
    grasp_hand_vid = hand_traj.landmarks[grasp_frame, GRASP_LANDMARK].copy()
    print(f"Hand position at grasp (middle finger MCP): {grasp_hand_vid}")

    # Use GRASP frame as reference for hand tracking
    init_hand_vid = grasp_hand_vid.copy()
    print(f"Using grasp frame as hand reference: {init_hand_vid}")

    # 3. Retarget Hand
    print("Retargeting hand...")
    retargeter = AdroitRetargeter()
    joint_traj = retargeter.retarget_sequence(hand_traj.landmarks)
    
    # 4. Simulation Loop
    env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    env.reset()
    
    # Get qpos indices
    model = env.unwrapped.model
    # Assumed structure: [30 hand, 7 object, 7 target]
    # Check joint names
    jnames = [model.joint(i).name for i in range(model.njnt)]
    # 'hand' joints usually start at 0. 'OBJTx' etc for object.
    
    # Calibration & Debug
    print("-" * 30)
    print("Calibration Data:")
    print(f"Default Reset QPos [0:3]: {env.unwrapped.data.qpos[0:3]}")
    
    # Calibration & Debug
    print("-" * 30)
    print("Calibration Data:")
    
    # Analyze computed ranges
    min_pos = joint_traj[:, 0:3].min(axis=0)
    max_pos = joint_traj[:, 0:3].max(axis=0)
    print(f"Computed Hand Pos Range: Min={min_pos}, Max={max_pos}")
    
    min_obj = obj_pos_sim.min(axis=0)
    max_obj = obj_pos_sim.max(axis=0)
    print(f"Computed Obj Pos Range: Min={min_obj}, Max={max_obj}")
    print("-" * 30)
    
    # Force a known good state for first frame to check camera
    # Reset puts hand above table
    env.reset()

    # Get model and data references
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Get ACTUAL object position from body xpos (not qpos which is joint-relative)
    obj_body_id = model.body("Object").id
    default_obj_pos = data.xpos[obj_body_id].copy()
    print(f"Default object pos (from xpos): {default_obj_pos}")

    # Get S_grasp site position - this is the hand's grasp point
    grasp_site_id = model.site("S_grasp").id
    grasp_site_pos = data.site_xpos[grasp_site_id].copy()
    print(f"S_grasp site pos: {grasp_site_pos}")

    palm_body_id = model.body("palm").id

    # Debug: Print hand geometry info
    print("\n--- HAND GEOMETRY DEBUG ---")

    # Print body names and positions
    print("Bodies:")
    for i in range(model.nbody):
        name = model.body(i).name
        pos = data.xpos[i]
        print(f"  {i}: {name} -> pos: {pos}")

    # Print site names if available
    print("\nSites:")
    for i in range(model.nsite):
        name = model.site(i).name
        pos = data.site_xpos[i]
        print(f"  {i}: {name} -> pos: {pos}")
    print("--- END DEBUG ---\n")

    # Object stays at default position
    obj_init_pos = default_obj_pos.copy()
    print(f"Object stays at: {obj_init_pos}")

    # Compute fingertip offset with grasp finger pose
    # Set root to origin, apply grasp frame finger angles, measure fingertip position
    data.qpos[:] = 0
    data.qpos[3:30] = joint_traj[grasp_frame, 3:]  # Grasp finger pose
    mujoco.mj_forward(model, data)
    mftip_id = model.site("S_mftip").id
    mftip_at_origin = data.site_xpos[mftip_id].copy()
    print(f"Fingertip when root at origin: {mftip_at_origin}")
    fingertip_offset = mftip_at_origin  # Offset from root to fingertip

    # Reset and recapture object position (reset changes it)
    env.reset()
    mujoco.mj_forward(model, data)
    obj_init_pos = data.xpos[obj_body_id].copy()
    print(f"Object after reset: {obj_init_pos}")

    # CALIBRATE: Find qpos that positions palm to grasp the object
    # Position palm ABOVE and BEHIND cube so fingers curl down onto it
    GRASP_OFFSET = np.array(grasp_offset)  # 8cm behind, 8cm up (fingers reach down)
    target_pos = obj_init_pos + GRASP_OFFSET
    print(f"Grasp target (with offset): {target_pos}")
    qpos_guess = np.array([0.0, 0.0, 0.0])

    for iteration in range(10):  # Iterative refinement
        data.qpos[0:3] = qpos_guess
        data.qpos[3:30] = joint_traj[grasp_frame, 3:]  # Use grasp finger pose
        mujoco.mj_forward(model, data)

        actual_palm = data.xpos[palm_body_id].copy()
        error = target_pos - actual_palm

        # Update qpos based on error (using known axis mapping)
        # palm_x = -qpos[0], palm_y = qpos[2] + offset, palm_z = qpos[1] + offset
        qpos_guess[0] -= error[0]  # X: qpos[0] moves palm in -X
        qpos_guess[2] += error[1]  # Y: qpos[2] moves palm in +Y
        qpos_guess[1] += error[2]  # Z: qpos[1] moves palm in +Z

        if np.linalg.norm(error) < 0.001:  # 1mm tolerance
            break

    grasp_qpos = qpos_guess.copy()
    print(f"Calibrated grasp qpos: {grasp_qpos}")
    print(f"Palm error after calibration: {error} (norm: {np.linalg.norm(error):.6f})")

    frame = env.render()
    if frame is None or np.mean(frame) < 1:
        print("CRITICAL: Default render is black! Camera issue?")
        cv2.imwrite("debug_black_frame.png", np.zeros((100,100)))
    else:
        cv2.imwrite("debug_reset_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Default render OK. Saved to debug_reset_frame.png")

    print(f"Rendering side-by-side replay to {output_path}...")
    height, width = 480, 480

    # Resize video frames to match sim
    vid_frames_resized = [cv2.resize(f, (width, height)) for f in frames]

    # Get object geom ID for contact detection
    try:
        obj_geom_id = model.geom("cube").id
        print(f"Object geom 'cube' ID: {obj_geom_id}")
    except:
        try:
            obj_geom_id = model.geom("Object").id  # Fallback name
            print(f"Object geom 'Object' ID: {obj_geom_id}")
        except:
            # List all geoms to find the object
            print("Could not find object geom. Available geoms:")
            for i in range(model.ngeom):
                print(f"  {i}: {model.geom(i).name}")
            obj_geom_id = None

    # Adjust cube size to match the Rubik's cube in the video
    if obj_geom_id is not None:
        current_size = model.geom_size[obj_geom_id].copy()
        print(f"Original cube size: {current_size}")
        # Increase cube size by 20% (slight increase)
        new_size = current_size * 1.2
        model.geom_size[obj_geom_id] = new_size
        print(f"New cube size: {new_size}")  # ~7.2cm cube

    # Debug: List fingertip geom IDs
    print("\nFingertip geom IDs:")
    finger_tips_debug = {
        'ff': 'C_ffdistal',
        'mf': 'C_mfdistal',
        'rf': 'C_rfdistal',
        'lf': 'C_lfdistal',
        'th': 'C_thdistal',
    }
    for name, geom_name in finger_tips_debug.items():
        try:
            gid = model.geom(geom_name).id
            print(f"  {name}: {geom_name} -> geom id {gid}")
        except Exception as e:
            print(f"  {name}: {geom_name} -> NOT FOUND: {e}")

    # Finger joint indices in qpos (flexion joints that close fingers)
    # Format: {finger: [J2_idx, J1_idx, J0_idx]} - MCP, PIP, DIP flexion
    FINGER_JOINTS = {
        'ff': [9, 10, 11],    # Index: FFJ2, FFJ1, FFJ0 (indices 9-11 in qpos[3:30] -> 12-14 in qpos)
        'mf': [13, 14, 15],   # Middle
        'rf': [17, 18, 19],   # Ring
        'lf': [22, 23, 24],   # Pinky (has extra J4 at 20)
        'th': [27, 28, 29],   # Thumb: THJ2, THJ1, THJ0
    }

    # Grasp state tracking
    grasp_complete = False  # True when all fingers are in contact
    GRIP_GAIN = 2.0  # Amplify finger closing (dialed back)
    CLOSE_RATE = 0.12  # How fast to close fingers before contact
    GRIP_FORCE_RATE = 0.05  # Additional closing after contact (gentle)
    BASE_FLEXION = 0.6  # Initial finger flexion - moderate pre-curl
    TRANSPORT_GRIP_BOOST = 0.08  # Extra grip force during transport

    # Fix NaN values in joint trajectory (from divide-by-zero in retargeting)
    joint_traj = np.nan_to_num(joint_traj, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clamp finger angles to reasonable range
    joint_traj[:, 3:] = np.clip(joint_traj[:, 3:], -1.5, 1.5)

    with VideoWriter(output_path, 30.0, width * 2, height) as writer:
        for i in range(num_frames):
            # Compute hand movement from grasp position (in video space)
            hand_pos = hand_traj.landmarks[i, GRASP_LANDMARK]
            hand_delta_x = hand_pos[0] - init_hand_vid[0]
            hand_delta_y = hand_pos[1] - init_hand_vid[1]
            hand_delta_z = hand_pos[2] - init_hand_vid[2]

            # Scale video movement to sim space
            SCALE_X, SCALE_Y, SCALE_Z = hand_scales

            # Apply delta to grasp_qpos (using same axis mapping as calibration)
            sim_qpos = grasp_qpos.copy()
            sim_qpos[0] -= hand_delta_x * SCALE_X  # X movement
            sim_qpos[2] -= hand_delta_y * SCALE_Y  # Y in video -> height in sim
            sim_qpos[1] += hand_delta_z * SCALE_Z  # Depth in video -> forward in sim

            sim_qpos[1] = np.clip(sim_qpos[1], -0.5, 0.5)
            sim_qpos[2] = np.clip(sim_qpos[2], -0.5, 0.8)

            joint_traj[i, 0:3] = sim_qpos

            # Apply GRIP_GAIN to finger flexion angles only (not thumb opposition/abduction)
            # Indices: 8-24 are finger joints, 25-29 are thumb
            joint_traj[i, 8:25] *= GRIP_GAIN  # Fingers only
            # Thumb: DON'T apply extra gain, just use computed values
            # joint_traj[i, 27:30] *= 1.5  # Disabled - was causing issues

            # Debug thumb at grasp frame
            if i == grasp_frame:
                print(f"\n--- THUMB DEBUG at frame {i} ---")
                print(f"THJ4 (opposition): {joint_traj[i, 25]:.3f}")
                print(f"THJ3 (abduction):  {joint_traj[i, 26]:.3f}")
                print(f"THJ2 (CMC flex):   {joint_traj[i, 27]:.3f}")
                print(f"THJ1 (MCP flex):   {joint_traj[i, 28]:.3f}")
                print(f"THJ0 (IP flex):    {joint_traj[i, 29]:.3f}")

            # Determine current phase label based on milestones
            current_phase = "IDLE"
            if milestones:
                sorted_ms = sorted(milestones, key=lambda x: x['frame'])
                for ms in sorted_ms:
                    if i >= ms['frame']:
                        current_phase = ms['label'].upper()
                    else:
                        break
            else:
                # Fallback to older 'phases' list if milestones not present
                for phase in phases:
                    if phase.start_frame <= i <= phase.end_frame:
                        current_phase = phase.label
                        break

            # Hybrid mode: kinematic before grasp, physics after
            if i < grasp_frame:
                # Kinematic mode - just compute positions, no physics
                env.unwrapped.data.qpos[0:30] = joint_traj[i]
                mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            else:
                # Physics mode - use smooth position control
                target_qpos = joint_traj[i].copy()
                current_qpos = env.unwrapped.data.qpos[0:30].copy()

                # Apply base flexion to fingers so they're pre-curled (NOT thumb - it uses negative values)
                for finger, joint_indices in FINGER_JOINTS.items():
                    if finger != 'th':  # Skip thumb - it has different sign convention
                        for jidx in joint_indices:
                            target_qpos[jidx] = max(target_qpos[jidx], BASE_FLEXION)

                # Contact-based finger closing during GRASP/TRANSPORT
                if current_phase in ["GRASP", "TRANSPORT"]:
                    contacts = get_finger_contacts(model, data, obj_geom_id)
                    all_in_contact = all(contacts.values())

                    # TRANSPORT needs extra grip to overcome gravity
                    grip_rate = GRIP_FORCE_RATE
                    if current_phase == "TRANSPORT":
                        grip_rate = GRIP_FORCE_RATE + TRANSPORT_GRIP_BOOST

                    # Debug: Print contact info every 20 frames in GRASP/TRANSPORT
                    if i >= grasp_frame and (i - grasp_frame) % 20 == 0:
                        print(f"\n--- CONTACT DEBUG Frame {i} ---")
                        print(f"Number of contacts: {data.ncon}")
                        print(f"Phase: {current_phase}, grip_rate: {grip_rate}")
                        cube_contacts = []
                        for ci in range(data.ncon):
                            c = data.contact[ci]
                            g1_name = model.geom(c.geom1).name if c.geom1 < model.ngeom else f"id{c.geom1}"
                            g2_name = model.geom(c.geom2).name if c.geom2 < model.ngeom else f"id{c.geom2}"
                            # Show all contacts involving cube
                            if c.geom1 == obj_geom_id or c.geom2 == obj_geom_id:
                                other_name = g1_name if c.geom2 == obj_geom_id else g2_name
                                cube_contacts.append(other_name if other_name else f"geom{c.geom1 if c.geom2 == obj_geom_id else c.geom2}")
                        print(f"All cube contacts: {cube_contacts}")
                        print(f"Finger contacts dict: {contacts}")

                    for finger, joint_indices in FINGER_JOINTS.items():
                        if finger == 'th':
                            # Thumb uses negative values for flexion
                            if contacts[finger]:
                                for jidx in joint_indices:
                                    target_qpos[jidx] = current_qpos[jidx] - grip_rate
                            else:
                                for jidx in joint_indices:
                                    target_qpos[jidx] = current_qpos[jidx] - CLOSE_RATE
                        else:
                            # Other fingers use positive values
                            if contacts[finger]:
                                for jidx in joint_indices:
                                    target_qpos[jidx] = current_qpos[jidx] + grip_rate
                            else:
                                for jidx in joint_indices:
                                    target_qpos[jidx] = current_qpos[jidx] + CLOSE_RATE

                    # Clamp finger angles to prevent over-extension
                    # Increased max flexion for tighter grip
                    for finger, joint_indices in FINGER_JOINTS.items():
                        if finger == 'th':
                            for jidx in joint_indices:
                                target_qpos[jidx] = np.clip(target_qpos[jidx], -2.5, 0.5)  # was -2.0
                        else:
                            for jidx in joint_indices:
                                target_qpos[jidx] = np.clip(target_qpos[jidx], -0.5, 2.5)  # was 2.0

                    if all_in_contact and not grasp_complete:
                        grasp_complete = True
                        print(f"  [Frame {i}] GRASP COMPLETE - all fingers in contact!")

                # Smooth interpolation (avoid sudden jumps)
                # Faster blend for fingers during grasp, slower arm during transport
                if current_phase == "TRANSPORT":
                    # Moderate fingers, slow arm during transport
                    alpha_arm = 0.2  # Slow arm movement
                    alpha_fingers = 0.35  # Moderate finger response
                    blended_qpos = current_qpos.copy()
                    blended_qpos[0:6] += alpha_arm * (target_qpos[0:6] - current_qpos[0:6])  # Arm
                    blended_qpos[6:30] += alpha_fingers * (target_qpos[6:30] - current_qpos[6:30])  # Fingers
                elif current_phase == "GRASP":
                    alpha = 0.3  # Gentler grasp
                    blended_qpos = current_qpos + alpha * (target_qpos - current_qpos)
                else:
                    alpha = 0.3
                    blended_qpos = current_qpos + alpha * (target_qpos - current_qpos)
                # Use actuator control instead of direct qpos (respects collisions)
                # Adroit uses position actuators - set ctrl to desired positions
                env.unwrapped.data.ctrl[:30] = blended_qpos

                # More substeps during grasp/transport for stable grip
                n_substeps = 10 if current_phase in ["GRASP", "TRANSPORT"] else 5
                for _ in range(n_substeps):
                    mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)

            # Reset grasp state on RELEASE
            if current_phase == "RELEASE":
                grasp_complete = False

            # Debug at grasp frame - see where things ACTUALLY end up
            if i == grasp_frame:
                actual_obj = data.xpos[obj_body_id]
                actual_palm = data.xpos[palm_body_id]
                print(f"\n--- GRASP FRAME {i} - ACTUAL POSITIONS ---")
                print(f"Object:    {actual_obj}")
                print(f"Palm:      {actual_palm}")
                print(f"Gap (palm-obj): {actual_palm - actual_obj}")
                print(f"--- END ---\n")

            # Render Sim
            sim_frame = env.render()
            sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)

            # Get Video Frame
            vid_frame = vid_frames_resized[i].copy()

            # --- Draw MediaPipe landmarks on video ---
            landmarks = hand_traj.landmarks[i]
            h_frame, w_frame = vid_frame.shape[:2]

            # MediaPipe hand connections
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),  # Thumb
                (0,5),(5,6),(6,7),(7,8),  # Index
                (0,9),(9,10),(10,11),(11,12),  # Middle
                (0,13),(13,14),(14,15),(15,16),  # Ring
                (0,17),(17,18),(18,19),(19,20),  # Pinky
                (5,9),(9,13),(13,17)  # Palm
            ]

            # Draw connections
            for conn in HAND_CONNECTIONS:
                pt1 = (int(landmarks[conn[0], 0] * w_frame), int(landmarks[conn[0], 1] * h_frame))
                pt2 = (int(landmarks[conn[1], 0] * w_frame), int(landmarks[conn[1], 1] * h_frame))
                cv2.line(vid_frame, pt1, pt2, (0, 255, 0), 2)

            # Draw landmark points
            for j, lm in enumerate(landmarks):
                x, y = int(lm[0] * w_frame), int(lm[1] * h_frame)
                color = (0, 0, 255) if j == GRASP_LANDMARK else (0, 255, 255)  # Red for tracking point
                cv2.circle(vid_frame, (x, y), 4, color, -1)

            # --- Draw object tracking box ---
            obj_x, obj_y = obj_pos_vid_norm[i]
            obj_cx, obj_cy = int(obj_x * w_frame), int(obj_y * h_frame)
            cv2.rectangle(vid_frame, (obj_cx - 30, obj_cy - 30), (obj_cx + 30, obj_cy + 30), (255, 0, 255), 2)
            cv2.circle(vid_frame, (obj_cx, obj_cy), 5, (255, 0, 255), -1)

            # --- Draw current phase (already computed above) ---
            # Phase color coding
            phase_colors = {
                "IDLE": (128, 128, 128),
                "APPROACH": (0, 255, 255),
                "GRASP": (0, 255, 0),
                "TRANSPORT": (255, 165, 0),
                "RELEASE": (255, 0, 0),
                "RETREAT": (128, 0, 128),
                "UNKNOWN": (255, 255, 255)
            }
            phase_color = phase_colors.get(current_phase, (255, 255, 255))

            # Draw phase label with background
            cv2.rectangle(vid_frame, (5, 5), (150, 40), (0, 0, 0), -1)
            cv2.putText(vid_frame, current_phase, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

            # Frame number on video
            cv2.putText(vid_frame, f"Frame: {i}", (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Overlay Info on Sim
            cv2.putText(sim_frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(sim_frame, current_phase, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)
            cv2.putText(sim_frame, f"S_Obj: {SCALE:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
            cv2.putText(sim_frame, f"S_Hand: {SCALE_X:.1f},{SCALE_Y:.1f},{SCALE_Z:.1f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            # Concatenate
            combined = np.hstack((vid_frame, sim_frame))
            
            writer.write_frame(combined)
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/pick-rubiks-cube.mp4")
    parser.add_argument("--out", default="adroit_replay_sbs.mp4") # Side-by-side
    parser.add_argument("--analysis", help="Path to Gemini analysis JSON")
    
    # Tuning params
    parser.add_argument("--obj-scale", type=float, default=0.8, help="Scale for object position mapping")
    parser.add_argument("--hand-scales", type=str, default="0.5,0.3,0.3", help="Scales for hand motion X,Y,Z (comma separated)")
    parser.add_argument("--grasp-offset", type=str, default="0.0,-0.08,0.08", help="Grasp target offset X,Y,Z (comma separated)")
    
    args = parser.parse_args()
    
    # Parse comma separated strings
    hand_scales = tuple(map(float, args.hand_scales.split(',')))
    grasp_offset = tuple(map(float, args.grasp_offset.split(',')))
    
    replay_m1(args.video, args.out, 
             obj_scale=args.obj_scale, 
             hand_scales=hand_scales, 
             grasp_offset=grasp_offset,
             analysis_path=args.analysis)
