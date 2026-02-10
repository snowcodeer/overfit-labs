"""
Gemini Video Analyzer

Uses Gemini 1.5 Pro to analyze manipulation videos and extract key events.
Input: Video Path
Output: JSON with {grasp_frame, release_frame, object_description}
"""

import os
import time
import argparse
import json
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    exit(1)

client = genai.Client(api_key=API_KEY)


def analyze_video(video_path: str):
    print(f"Analyzing video: {video_path}")

    # 1. Upload Video
    print("Uploading to Gemini...")
    # The google.genai SDK uses 'file' parameter, not 'path'
    video_file = client.files.upload(file=video_path)
    print(f"Upload complete: {video_file.name}")
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")
        
    print("\nVideo processed.")
    
    # 2. Prompt for Analysis
    model_name = "models/gemini-3-flash-preview"
    print(f"Using model: {model_name}", flush=True)
    
    prompt = """
    Analyze this video of a hand manipulating an object. Watch the ENTIRE video carefully.

    IMPORTANT: I need ACCURATE frame numbers, not placeholders. The video runs at 30 FPS.

    Steps:
    1. Watch the full video to understand the duration and action
    2. Identify what type of task this is (pick, throw, catch, push, slide, rotate)
    3. Find the KEY MOMENTS and estimate their frame numbers

    Focus on the RIGHT HAND if multiple hands are visible.

    Return ONLY a JSON object:
    {
        "object_name": "name of the object being manipulated",
        "object_description": "brief description of the object",
        "task_type": "pick|throw|catch|push|slide|rotate",
        "milestones": [
            {"label": "event_name", "frame": <frame_number>, "description": "what happens"}
        ],
        "eval_code": {
            "success_criteria": [
                {"name": "is_grasped", "description": "Hand has grip on object", "condition": "n_contacts >= 3"},
                {"name": "is_lifted", "description": "Object above table", "condition": "obj_height > 0.05"}
            ],
            "metrics": [
                {"name": "distance_to_goal", "description": "Distance to target", "formula": "np.linalg.norm(obj_pos - target_pos)"},
                {"name": "grasp_quality", "description": "Contact points", "formula": "min(n_contacts, 5)"}
            ]
        }
    }

    EVAL CODE - Based on task_type, generate success_criteria and metrics:
    - pick: is_grasped, is_lifted, is_at_target + distance_to_goal, lift_height
    - throw: is_released, reached_apex + throw_velocity, flight_distance
    - catch: is_caught, is_stabilized + catch_timing
    - slide: is_contacted, reached_target + slide_distance
    - rotate: rotation_started, target_angle_reached + rotation_angle

    Common milestone patterns:
    - pick: approach, grasp, lift, transport, release
    - throw: hold, wind_up, release, flight
    - catch: prepare, catch, stabilize
    """
    
    print("Requesting analysis...")
    response = client.models.generate_content(
        model=model_name,
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    print("Analysis complete.")
    # print(response.text) # Reduce noise
    
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
    
    # Remove "json" prefix if it exists after splitting
    if text.startswith("json"):
        text = text[4:].strip()

    return json.loads(text)


def generate_eval_code(analysis: dict, task_name: str) -> str:
    """Generate eval.py content from Gemini analysis."""
    task_type = analysis.get("task_type", "pick")
    eval_spec = analysis.get("eval_code", {})

    success_criteria = eval_spec.get("success_criteria", [])
    metrics = eval_spec.get("metrics", [])

    # Default criteria if not provided
    if not success_criteria:
        success_criteria = [
            {"name": "is_grasped", "description": "Hand has grip on object", "condition": "n_contacts >= 3"},
            {"name": "is_lifted", "description": "Object above table", "condition": "obj_height > 0.05"},
            {"name": "is_at_target", "description": "Object near target", "condition": "dist_to_target < 0.1"}
        ]
    if not metrics:
        metrics = [
            {"name": "distance_to_goal", "description": "Distance to target", "formula": "np.linalg.norm(obj_pos - target_pos)"},
            {"name": "grasp_quality", "description": "Contact points", "formula": "min(n_contacts, 5)"},
            {"name": "lift_height", "description": "Object height", "formula": "obj_pos[2]"}
        ]

    # Build success functions
    success_fns = ""
    success_list = []
    for c in success_criteria:
        name, desc, cond = c["name"], c["description"], c["condition"]
        success_fns += f'''
def {name}(env) -> bool:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body("Object").id]
    n_contacts = data.ncon
    obj_height = obj_pos[2]
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    dist_to_target = np.linalg.norm(obj_pos - target_pos)
    return {cond}
'''
        success_list.append(f'("{name.replace("is_", "")}", {name})')

    # Build metric functions
    metric_fns = ""
    metric_list = []
    for m in metrics:
        name, desc, formula = m["name"], m["description"], m["formula"]
        metric_fns += f'''
def compute_{name}(env) -> float:
    """{desc}"""
    data = env.unwrapped.data
    model = env.unwrapped.model
    obj_pos = data.xpos[model.body("Object").id]
    n_contacts = data.ncon
    target_pos = getattr(env, 'target_pos', np.array([0.3, 0.0, 0.2]))
    return float({formula})
'''
        metric_list.append(f'("{name}", compute_{name})')

    return f'''"""
Evaluation Functions for: {task_name}
Task Type: {task_type}
Auto-generated by Gemini Video Analysis
"""

import numpy as np
from typing import Dict, Tuple, Any

# ============ SUCCESS CRITERIA ============
{success_fns}

# ============ METRICS ============
{metric_fns}

# ============ EVALUATION SUITE ============

class EvalSuite:
    """Complete evaluation suite for this task."""

    def __init__(self, env):
        self.env = env
        self.success_checks = [
            {", ".join(success_list)}
        ]
        self.metric_fns = [
            {", ".join(metric_list)}
        ]

    def check_success(self) -> Tuple[bool, Dict[str, bool]]:
        results = {{}}
        for name, fn in self.success_checks:
            try:
                results[name] = fn(self.env)
            except:
                results[name] = False
        return all(results.values()), results

    def compute_metrics(self) -> Dict[str, float]:
        metrics = {{}}
        for name, fn in self.metric_fns:
            try:
                metrics[name] = fn(self.env)
            except:
                metrics[name] = float('nan')
        return metrics

    def evaluate_step(self) -> Dict[str, Any]:
        success, milestone_progress = self.check_success()
        metrics = self.compute_metrics()
        return {{"success": success, "metrics": metrics, "milestone_progress": milestone_progress}}
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="video_analysis.json", help="Path to save JSON output")
    parser.add_argument("--eval-output", default=None, help="Path to save eval.py")

    args = parser.parse_args()

    try:
        result = analyze_video(args.video)

        with open(args.output, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Saved analysis to {args.output}")

        # Generate eval.py
        eval_output = args.eval_output or (Path(args.output).parent / "eval.py")
        task_name = Path(args.output).parent.name
        eval_code = generate_eval_code(result, task_name)
        with open(eval_output, "w") as f:
            f.write(eval_code)
        print(f"Saved eval code to {eval_output}")

    except Exception as e:
        print(f"Error: {e}")
