
import os
import argparse
import time
import json
import re
import subprocess
import sys
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Setup API Key
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables.")
    return genai.Client(api_key=api_key)

def upload_to_gemini(client, path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = client.files.upload(file=path, config=types.UploadFileConfig(mime_type=mime_type))
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def wait_for_files_active(client, files):
    """Waits for the given files to be active."""
    print("Waiting for file processing...", end="")
    for name in (file.name for file in files):
        file = client.files.get(name=name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(2)
            file = client.files.get(name=name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")

def run_replay(video_path, obj_scale, hand_scales, grasp_offset, output_path="adroit_replay_sbs.mp4"):
    """Runs the replay_m1.py script with the given parameters."""
    cmd = [
        sys.executable, "scripts/replay_m1.py",
        "--video", video_path,
        "--out", output_path,
        "--obj-scale", str(obj_scale),
        "--hand-scales", f"{hand_scales[0]},{hand_scales[1]},{hand_scales[2]}",
        "--grasp-offset", f"{grasp_offset[0]},{grasp_offset[1]},{grasp_offset[2]}"
    ]
    print(f"Running replay with: {cmd}")
    subprocess.run(cmd, check=True)
    return output_path

class GeminiCritiqueAgent:
    def __init__(self, client, model_name="models/gemini-1.5-flash"):
        self.client = client
        self.model_name = model_name
        self.generation_config = types.GenerateContentConfig(
            temperature=0.4,
            top_p=0.95,
            top_k=64,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )
        self.system_prompt = """
        You are a Robotics Sim-to-Real Expert.
        
        Goal: Align a MuJoCo simulation (Right side of video) with a Real-world video (Left side).
        The simulation is driven by the real video's trajectory, but due to calibration errors, the hand might not reach the object correctly, or the object size might be wrong.
        
        You have control over 3 parameters:
        1. `obj_scale`: Scales the object position mapping (Video -> Sim). 
           - Increase if the robot hand goes *past* the object on all axes.
           - Decrease if the robot hand falls short.
        2. `hand_scales` (x, y, z): Scales the hand's motion delta. 
           - X: Left/Right motion width.
           - Y: Vertical motion height (Video Y -> Sim Z).
           - Z: Depth/Forward motion (Video Depth -> Sim Y).
        3. `grasp_offset` (x, y, z): A fixed offset added to the target grasp position.
           - Use this to fix constant systematic errors (e.g., hand always too high or too far right).
           
        Process:
        1. Watch the side-by-side video.
        2. Observe the GRASP moment. Does the virtual hand (Right) touch the virtual object at the same time the real hand (Left) touches the real object?
        3. Observe the TRANSPORT. Does the object stay attached to the hand?
        4. Suggest updated parameters to fix any alignment errors.
        
        Output JSON Format:
        {
            "analysis": "Describe what you see. E.g., 'The hand grasps too high above the cube.'",
            "satisfied": false,
            "new_obj_scale": 0.8,
            "new_hand_scales": [0.5, 0.3, 0.3],
            "new_grasp_offset": [0.0, -0.08, 0.08]
        }
        
        Constraints:
        - Make incremental changes (e.g., +/- 0.02 for offsets, +/- 0.05 for scales).
        - If the grasp looks good and the object is lifted successfully, set "satisfied": true.
        - `grasp_offset` Y is usually negative because the palm is above the object.
        """

    def analyze(self, video_file, current_params):
        prompt = f"""
        Current Parameters:
        - obj_scale: {current_params['obj_scale']}
        - hand_scales: {current_params['hand_scales']}
        - grasp_offset: {current_params['grasp_offset']}
        
        Analyze the video and output the JSON.
        """
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[video_file, prompt],
            config=self.generation_config.model_copy(update={"system_instruction": self.system_prompt}),
        )
        print(f"Gemini Response: {response.text}")
        return json.loads(response.text)

def main():
    client = setup_gemini()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/pick-rubiks-cube.mp4")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    
    # Initial Params (Good defaults)
    current_params = {
        "obj_scale": 0.8,
        "hand_scales": [0.5, 0.3, 0.3],
        "grasp_offset": [0.0, -0.08, 0.08]
    }
    
    agent = GeminiCritiqueAgent(client, model_name="models/gemini-2.0-flash-exp") # Fast, multimodal
    
    for i in range(args.iterations):
        print(f"\n=== Iteration {i+1}/{args.iterations} ===")
        
        # 1. Run Replay
        output_video = f"replay_iter_{i}.mp4"
        run_replay(
            args.video, 
            current_params['obj_scale'], 
            current_params['hand_scales'], 
            current_params['grasp_offset'],
            output_path=output_video
        )
        
        # 2. Upload Video
        print("Uploading to Gemini...")
        video_file = upload_to_gemini(client, output_video, mime_type="video/mp4")
        wait_for_files_active(client, [video_file])
        
        # 3. Get Critique
        result = agent.analyze(video_file, current_params)
        
        # 4. Update Params
        if result.get("satisfied", False):
            print("Gemini is satisfied! Stopping loop.")
            break
            
        current_params["obj_scale"] = result.get("new_obj_scale", current_params["obj_scale"])
        current_params["hand_scales"] = result.get("new_hand_scales", current_params["hand_scales"])
        current_params["grasp_offset"] = result.get("new_grasp_offset", current_params["grasp_offset"])
        
        print(f"Updated Params for next run: {current_params}")
        
    # Save optimized config
    with open("optimized_sim_params.json", "w") as f:
        json.dump(current_params, f, indent=4)
        print("Saved optimized parameters to optimized_sim_params.json")

if __name__ == "__main__":
    main()
