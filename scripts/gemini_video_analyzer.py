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

import google.generativeai as genai
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("ERROR: GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=API_KEY)


def analyze_video(video_path: str):
    print(f"Analyzing video: {video_path}")
    
    # 1. Upload Video
    print("Uploading to Gemini...")
    video_file = genai.upload_file(path=video_path)
    print(f"Upload complete: {video_file.name}")
    
    # Wait for processing
    while video_file.state.name == "PROCESSING":
        print('.', end='', flush=True)
        time.sleep(2)
        video_file = genai.get_file(video_file.name)
        
    if video_file.state.name == "FAILED":
        raise ValueError("Video processing failed.")
        
    print("\nVideo processed.")
    
    # 2. Prompt for Analysis
    model_name = "models/gemini-3-flash-preview"
    print(f"Using model: {model_name}", flush=True)
    model = genai.GenerativeModel(model_name=model_name)
    
    prompt = """
    Analyze this video of a robotic hand (or human hand) manipulating an object.
    Focus primarily on the **Right Hand** for all tracking and event timing if multiple hands are present.
    
    I need to extract the exact frame numbers for key events to train a robot.
    The video is 30 FPS.
    
    Please identify the task (e.g., pick_and_place, throw, rotate, push) and extract the list of critical keyframes (milestones) for that specific task.
    Examples:
    - For 'pick': [grasp_frame, lift_frame, release_frame]
    - For 'throw': [hold_frame, swing_start_frame, release_frame, flight_end_frame]
    - For 'push': [contact_frame, move_start_frame, move_end_frame]
    
    Return ONLY a JSON object with this structure:
    {
        "object_name": "string",
        "object_description": "string",
        "task_type": "string",
        "milestones": [
            {"label": "string (human readable event name)", "frame": int, "description": "short explanation"}
        ]
    }
    """
    
    print("Requesting analysis...")
    response = model.generate_content(
        [video_file, prompt],
        generation_config={"response_mime_type": "application/json"}
    )
    
    print("Analysis complete.")
    print(response.text)
    
    # Clean up (optional, files auto-expire)
    # genai.delete_file(video_file.name)
    
    return json.loads(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--output", default="video_analysis.json", help="Path to save JSON output")
    
    args = parser.parse_args()
    
    try:
        result = analyze_video(args.video)
        
        with open(args.output, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"Saved analysis to {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
