import sys
import os
import configparser
import random
import cv2
import librosa
import numpy as np
import yt_dlp
import subprocess
import google.generativeai as genai
import time
import shutil
from instagrapi import Client
from typing import Dict, List, Tuple
from pathlib import Path # Import Path for instagrapi


# --- DEBUG / ENCODING FIXES (add these at the very top) ---
print("--- Executing main.py version 2024-06-26-FINAL ---")
sys.stdout.flush() # Ensure this prints immediately

# Force UTF-8 encoding for stdout/stderr if not already set
# This helps prevent 'charmap' codec errors with console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
if sys.stderr.encoding != 'utf-8':
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
# Also set environment variable as a belt-and-braces approach for subprocesses
os.environ['PYTHONIOENCODING'] = 'utf-8'
# -------------------------------------------------------------


# ==============================================================================
# PHASE 0: CONFIGURATION AND SETUP (LOCAL-FIRST DESIGN)
# ==============================================================================
def setup_configuration() -> Dict:
    """Loads all settings and secrets directly from the config.ini file."""
    print("PHASE 0: Loading configuration from config.ini...")
    config = configparser.ConfigParser()
    config_file = 'config.ini'
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"CRITICAL: '{config_file}' not found.")
    config.read(config_file)

    secrets = config['SECRETS']
    settings = config['SETTINGS']

    # Directly return a dictionary of all configs
    config_data = {
        "instagram_username": secrets.get('INSTAGRAM_USERNAME'),
        "instagram_password": secrets.get('INSTAGRAM_PASSWORD'),
        "gemini_api_key": secrets.get('GEMINI_API_KEY'),
        "sources": [url.strip() for url in settings.get('YOUTUBE_SOURCES', '').split(',') if url.strip()],
        "upload_mode": settings.get('UPLOAD_MODE', 'best_part').lower(),
        # Retrieve ffmpeg_path from config.ini, defaulting to the hardcoded path if not found.
        # This allows config.ini to override the default derived path.
        "ffmpeg_path": settings.get('FFMPEG_PATH', os.path.join(os.getcwd(), 'ffmpeg', 'bin')),
        "starting_part": int(settings.get('STARTING_PART', '1')),
        "num_parts_to_upload": int(settings.get('NUM_PARTS_TO_UPLOAD', '1')),
        "clip_duration": int(settings.get('CLIP_DURATION', '80'))  # Default to 80 seconds if not set
    }

    if 'your_instagram_username' in config_data['instagram_username']:
        raise ValueError("CRITICAL: Please update your credentials in config.ini.")
    if not config_data['sources']:
        raise ValueError("CRITICAL: Please add at least one URL to YOUTUBE_SOURCES in config.ini.")
    if not config_data['gemini_api_key'] or 'YOUR_GEMINI_API_KEY' in config_data['gemini_api_key']:
        raise ValueError("CRITICAL: Please provide your Gemini API Key in config.ini.")

    # Pylance false positive: 'configure' is not exported. It is, via the alias.
    genai.configure(api_key=config_data['gemini_api_key']) # type: ignore
    return config_data

# ==============================================================================
# PHASE 1: VIRAL SCOUTING (INTELLIGENT SOURCE HANDLING)
# ==============================================================================

def get_source_video(sources: List[str], ffmpeg_path: str) -> Tuple[str, str, str]:
    """
    Selects the first source and prepares it for processing.
    Handles YouTube URLs (single/playlist) and local file paths.
    Returns the path to the prepared video, its title, and the original source URL/path.
    """
    print("PHASE 1.1: Selecting the first source from the list...")
    if not sources:
        raise ValueError("CRITICAL: The YOUTUBE_SOURCES list is empty in config.ini.")
    source_path_str = sources[0]  # Select the first source (string)
    video_title = "Unknown Title"

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    source_video_path = os.path.join(temp_dir, "source_video.mp4")

    # --- INTELLIGENT SOURCE DETECTION ---
    if source_path_str.lower().startswith('http'):
        # --- This is a YOUTUBE URL ---
        print(f"Source is a YouTube URL. Preparing for download...")
        video_url = None

        ydl_common_opts = {'quiet': True, 'ffmpeg_location': ffmpeg_path}

        if "playlist?list=" in source_path_str:
            # This is a PLAYLIST
            print(f"Source is a playlist. Selecting a random video from: {source_path_str}")
            with yt_dlp.YoutubeDL({'extract_flat': 'in_playlist', **ydl_common_opts}) as ydl:
                info = ydl.extract_info(source_path_str, download=False)
                if not info: # Robust check for info extraction failure
                    raise ValueError(f"Failed to extract info from playlist: {source_path_str}")
                if not info.get('entries'): # Robust check for empty playlist
                    raise ValueError(f"No videos found in the playlist: {source_path_str}")
                video_info = random.choice(info['entries'])
                video_url = video_info.get('url')
                video_title = video_info.get('title', video_title)
        else:
            # This is a SINGLE VIDEO
            print(f"Source is a single video: {source_path_str}")
            video_url = source_path_str
            with yt_dlp.YoutubeDL(ydl_common_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if not info: # Robust check for info extraction failure
                    raise ValueError(f"Failed to extract info from video: {video_url}")
                video_title = info.get('title', video_title)

        if not video_url: raise ValueError("Failed to identify a valid video URL from the YouTube source.")

        print(f"Selected video: '{video_title}'")
        print("PHASE 1.2: Downloading video for analysis...")
        
        ydl_download_opts = {
            'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': source_video_path,
            'quiet': False, # Set to True for less verbose output during download
            'merge_output_format': 'mp4',
            'overwrites': True,
            'ffmpeg_location': ffmpeg_path, # Explicitly pass ffmpeg path for merging
        }
        with yt_dlp.YoutubeDL(ydl_download_opts) as ydl:
            ydl.download([video_url])
    
    else:
        # --- This is a LOCAL FILE PATH ---
        print(f"Source is a local file path: {source_path_str}")
        if not os.path.exists(source_path_str):
            raise FileNotFoundError(f"Local video file not found at: {source_path_str}")
        
        print("PHASE 1.2: Copying local video for analysis...")
        shutil.copy(source_path_str, source_video_path)
        
        video_title = os.path.splitext(os.path.basename(source_path_str))[0]
        print(f"Selected video: '{video_title}'")

    if not os.path.exists(source_video_path):
        raise FileNotFoundError(f"Source video could not be prepared at: {source_video_path}")

    return source_video_path, video_title, source_path_str

# ==============================================================================
# PHASE 2: STUDIO PRODUCTION (WITH COLOR GRADING)
# ==============================================================================

def process_video_clip(input_path: str, output_path: str, start_time: float, duration: int, ffmpeg_path: str):
    """
    Cuts and reformats the video, ensuring the local FFmpeg path is used.
    """
    print("PHASE 2: Processing clip with FFmpeg (16:9 Padded Format)...")
    
    ffmpeg_exe_name = 'ffmpeg.exe' if os.name == 'nt' else 'ffmpeg'
    ffmpeg_exe_path = os.path.join(ffmpeg_path, ffmpeg_exe_name)

    if not os.path.exists(ffmpeg_exe_path):
        raise FileNotFoundError(f"FFmpeg executable not found at: {ffmpeg_exe_path}. "
                                f"Please ensure FFmpeg is downloaded and extracted into the correct directory specified in config.ini.")

    command = [
        ffmpeg_exe_path, # Use the full path to the executable
        '-ss', str(start_time),
        '-i', input_path,
        '-t', str(duration),
        '-vf', "scale=-2:1080,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=white", # Pad to 1920x1080 (16:9)
        '-c:a', 'aac',
        '-y', output_path
    ]

    try:
        run_env = os.environ.copy()
        # Prepend our ffmpeg_path to the existing PATH
        run_env["PATH"] = f"{ffmpeg_path}{os.pathsep}{run_env.get('PATH', '')}"
        
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL, # Suppress standard output
            stderr=subprocess.PIPE,    # Capture standard error for debugging
            env=run_env                # Pass the custom environment here
        )
        print(f"Final 16:9 clip created at '{output_path}'")

    except subprocess.CalledProcessError as e:
        print("--- FFmpeg Error ---")
        print(f"Command: {' '.join(command)}")
        print(f"Return Code: {e.returncode}")
        print(f"Error Output:\n{e.stderr.decode()}")
        raise
    except FileNotFoundError:
        print(f"Error: FFmpeg command not found. Please ensure FFmpeg is installed and accessible at {ffmpeg_exe_path}.")
        raise

# ==============================================================================
# ALL OTHER FUNCTIONS (find_golden_segment, generate_caption, upload)
# ==============================================================================

def find_golden_segment(video_path: str, clip_duration: int = 120) -> float:
    """Analyzes video and audio with detailed logging to find the most intense segment."""
    print("PHASE 1.3: Analyzing video for 'Golden Segment'...")
    
    # --- Motion Analysis ---
    print("Analyzing motion with OpenCV...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file for analysis: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: 
        print("Warning: FPS is 0, defaulting to 30.")
        fps = 30 # Default to 30 FPS if not available

    motion_scores = []
    ret, prev_frame = cap.read()
    if not ret: 
        cap.release()
        raise ValueError("Could not read the first frame for motion analysis.")
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_scores.append(np.sum(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray
    cap.release()

    if not motion_scores:
        print("Warning: No motion data collected. Video might be too short or corrupted.")
        return 0.0 # Default to start time 0 if no motion data

    # --- Audio Analysis ---
    print("Analyzing audio with Librosa...")
    try:
        y, sr = librosa.load(video_path, sr=None) # sr=None loads original sample rate
        rms = librosa.feature.rms(y=y)[0]
        # Resample RMS scores to match the number of video frames
        audio_scores_resampled = np.interp(
            np.linspace(0, len(rms) - 1, len(motion_scores)),
            np.arange(len(rms)),
            rms
        )
        audio_scores = audio_scores_resampled
    except Exception as e:
        print(f"Warning: Could not process audio ({e}). Proceeding with motion analysis only.")
        audio_scores = np.zeros(len(motion_scores))

    # --- Combined Analysis ---
    print("Finding highest-impact window...")
    motion_scores = np.array(motion_scores)
    audio_scores = np.array(audio_scores)
    
    # Normalize scores to a 0-1 range
    norm_motion = (motion_scores - np.min(motion_scores)) / (np.max(motion_scores) - np.min(motion_scores) + 1e-6)
    norm_audio = (audio_scores - np.min(audio_scores)) / (np.max(audio_scores) - np.min(audio_scores) + 1e-6)
    combined_scores = norm_motion + norm_audio

    # Ensure window_size doesn't exceed the number of available frames
    window_size = int(fps * clip_duration)
    if window_size <= 0: # Ensure window_size is at least 1
        window_size = 1
    
    if len(combined_scores) < window_size:
        print(f"Warning: Video ({len(combined_scores)/fps:.2f}s) is shorter than desired clip duration ({clip_duration}s). Using full video.")
        return 0.0 # Start from beginning if video is too short
    
    max_score = -1
    best_start_frame = 0
    
    # Iterate to find the best segment
    for i in range(len(combined_scores) - window_size + 1):
        current_score = np.sum(combined_scores[i:i+window_size])
        if current_score > max_score:
            max_score, best_start_frame = current_score, i
            
    start_time = best_start_frame / fps
    
    print(f"Found 'Golden Segment' starting at {start_time:.2f} seconds.")
    return start_time

def generate_caption_with_gemini(movie_title: str) -> str:
    """Generates a viral Instagram caption using the Google Gemini API."""
    print("PHASE 3: Generating caption with Gemini API...")
    try:
        # Pylance false positive: 'GenerativeModel' is not exported. It is, via the alias.
        model = genai.GenerativeModel('gemini-1.5-flash-latest') # type: ignore
        
        prompt = f"""
        You are a viral Instagram strategist for a movie clips account. Your tone is cool, insightful, and engaging.
        Create a caption for a clip from '{movie_title}'.
        The caption must have three parts:
        1. A very short, punchy, scroll-stopping hook (max 10 words).
        2. A relatable, one-sentence observation about the scene's theme (e.g., power, betrayal, motivation).
        3. A simple, open-ended question to drive comments.
        Directly after the caption, provide 15 relevant hashtags, mixing broad tags (#movieclips, #cinema) with specific ones related to the movie, actors, or theme. Do not write the word 'Hashtags:'.
        """
        # Pylance false positive: 'types' is not exported. It is, via the alias.
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(temperature=0.8)) # type: ignore
        generated_caption = response.text.strip()
        print("Caption generated successfully.")
        return generated_caption
    except Exception as e:
        print(f"Gemini API Error: {e}. Using fallback caption.")
        return f"An incredible scene from {movie_title}!\n\n#movieclips #cinema #{movie_title.replace(' ', '').lower()}"

def upload_to_instagram(video_path_str: str, caption: str, username: str, password: str):
    print("FINAL STEP: Uploading to Instagram as a Reel...")
    cl = Client()
    
    # Convert string paths to Path objects for instagrapi functions
    session_file_path = Path(f"session-{username}.json")
    video_path_obj = Path(video_path_str)

    # Robust login attempt
    try:
        if session_file_path.exists():
            cl.load_settings(session_file_path) # Pass Path object
            print("Loaded session from file.")
        
        # Check if already logged in by inspecting cl.user_id, which is set after successful authentication
        if not cl.user_id: # cl.user_id will be None if not logged in
            print("Not logged in. Attempting login...")
            cl.login(username, password)
            cl.dump_settings(session_file_path) # Pass Path object
            print("Login successful and session saved.")
        else:
            print("Already logged in from session file.")

    except Exception as e:
        print(f"Instagram login failed: {e}")
        # Only remove session file if it's a login failure, not upload failure
        if session_file_path.exists():
            session_file_path.unlink() # Use Path.unlink()
            print("Removed potentially stale session file after login error.")
        raise # Re-raise the exception to propagate the login error

    print(f"Uploading '{video_path_str}'...")
    try:
        cl.clip_upload(path=video_path_obj, caption=caption) # Pass Path object
        print("Upload successful!")
    except Exception as e: # type: ignore # Pylance false positive: This block is reachable for upload errors.
        print(f"An error occurred during upload: {e}")
        # For an upload error, still remove session file for robust cleanup and re-login on next run
        if session_file_path.exists():
            session_file_path.unlink()
            print("Deleted potentially stale session file after upload error.")
        raise # Re-raise the exception to propagate the upload error

def update_starting_part_in_config(new_value: int, config_file: str = 'config.ini'):
    """Updates the STARTING_PART value in the config.ini file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'SETTINGS' not in config:
        config['SETTINGS'] = {} # Ensure section exists if it somehow doesn't
    config['SETTINGS']['STARTING_PART'] = str(new_value)
    with open(config_file, 'w') as f:
        config.write(f)
    print(f"Updated STARTING_PART in {config_file} to {new_value}")

def remove_source_from_config(source_to_remove: str, config_file: str = 'config.ini'):
    """Removes a specific source URL from the YOUTUBE_SOURCES in config.ini."""
    print(f"Attempting to remove '{source_to_remove}' from {config_file}...")
    config = configparser.ConfigParser()
    config.read(config_file)
    if 'SETTINGS' not in config:
        config['SETTINGS'] = {} # Ensure section exists
    
    current_sources = [url.strip() for url in config['SETTINGS'].get('YOUTUBE_SOURCES', '').split(',') if url.strip()]
    
    if source_to_remove in current_sources:
        sources_list = [s for s in current_sources if s != source_to_remove] # Rebuild list to avoid issues with remove()
        config['SETTINGS']['YOUTUBE_SOURCES'] = ','.join(sources_list)
        with open(config_file, 'w') as f:
            config.write(f)
        print(f"Successfully removed '{source_to_remove}' from YOUTUBE_SOURCES.")
    else:
        print(f"Warning: Source '{source_to_remove}' not found in YOUTUBE_SOURCES. No changes made.")

# ==============================================================================
# MAIN ORCHESTRATOR (LOCAL-FIRST DESIGN)
# ==============================================================================
if __name__ == "__main__":
    temp_dir = "temp"
    try:
        config = setup_configuration()
        
        source_video, video_title, processed_source_url = get_source_video(config['sources'], config['ffmpeg_path'])
        
        if config['upload_mode'] == 'best_part':
            print("\n--- Running in 'best_part' mode ---")
            clip_duration = config['clip_duration']
            start_time = find_golden_segment(source_video, clip_duration=clip_duration)
            final_clip = os.path.join(temp_dir, "final_clip.mp4")
            process_video_clip(source_video, final_clip, start_time, clip_duration, config['ffmpeg_path'])
            final_caption = generate_caption_with_gemini(video_title)
            upload_to_instagram(final_clip, final_caption, config['instagram_username'], config['instagram_password'])
        
        elif config['upload_mode'] == 'all_parts':
            print("\n--- Running in 'all_parts' mode ---")
            cap = cv2.VideoCapture(source_video)
            if not cap.isOpened():
                raise ValueError(f"Could not open source video for duration calculation: {source_video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            video_duration = video_duration_frames / fps if fps > 0 else 0 # Handle 0 FPS
            cap.release()
            
            if video_duration == 0:
                raise ValueError(f"Could not determine video duration for '{source_video}'. Check video file integrity.")

            clip_duration = config['clip_duration']
            total_parts = int(np.ceil(video_duration / clip_duration))
            print(f"Video will be split into {total_parts} parts of up to {clip_duration}s each.")
            
            start_part_num = config['starting_part']
            num_parts_to_process = config['num_parts_to_upload']
            
            processing_start_idx = start_part_num - 1 # Convert to 0-based index
            processing_end_idx = min(processing_start_idx + num_parts_to_process, total_parts)
            
            print(f"Processing parts from {processing_start_idx + 1} to {processing_end_idx}.")

            for i in range(processing_start_idx, processing_end_idx):
                part_num = i + 1
                start_time = i * clip_duration
                
                # Adjust duration for the last part if it's shorter
                current_clip_duration = min(clip_duration, video_duration - start_time)
                if current_clip_duration <= 0.1: # Allow a small buffer for very short last clips
                    print(f"Part {part_num} would be empty or too short ({current_clip_duration:.2f}s), skipping.")
                    continue

                print(f"\n--- Processing Part {part_num}/{total_parts} (Start: {start_time:.2f}s, Duration: {current_clip_duration:.2f}s) ---")
                final_clip_part = os.path.join(temp_dir, f"final_clip_part_{part_num}.mp4")
                
                process_video_clip(source_video, final_clip_part, start_time, current_clip_duration, config['ffmpeg_path'])
                
                caption_base = generate_caption_with_gemini(video_title)
                final_caption = f"PART {part_num}/{total_parts}\n\n{caption_base}"
                
                upload_to_instagram(final_clip_part, final_caption, config['instagram_username'], config['instagram_password'])
                
                # --- Update STARTING_PART in config.ini after each upload ---
                # Set starting_part for the *next* run
                update_starting_part_in_config(part_num + 1)
                
                # Check if this was the last part of the *current processing batch*
                if part_num < processing_end_idx: # If there are more parts in this batch
                    wait_time = random.randint(5 * 60, 10 * 60) # 5 to 10 minutes
                    print(f"Upload complete. Waiting for {wait_time / 60:.0f} minutes (randomized) before posting next part...")
                    time.sleep(wait_time)
            
            # --- After the loop, check if the whole video has been processed ---
            if processing_end_idx == total_parts:
                print(f"\n--- All {total_parts} parts for '{video_title}' have been completed ---")
                remove_source_from_config(processed_source_url)
                update_starting_part_in_config(1)  # Reset for the next video in the source list
            else:
                print(f"\n--- Processed parts {processing_start_idx + 1} to {processing_end_idx} for '{video_title}'. More parts remain. ---")

        else:
            raise ValueError(f"Invalid UPLOAD_MODE '{config['upload_mode']}' in config.ini. Must be 'best_part' or 'all_parts'.")
            
    except Exception as e:
        print(f"\nFATAL ERROR during the process: {e}")
    finally:
        print("Cleaning up temporary files...")
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print("Temporary directory 'temp' removed.")
            except OSError as e:
                print(f"Error removing temporary directory: {e}. Please remove it manually if it persists.")
        print("Cleanup complete.")
import time
print("Sleeping for 12 hours before next run...")
time.sleep(43200)  # 12 hours = 43200 seconds
os.execv(sys.executable, ['python'] + sys.argv)  # restart the script

        #,https://youtu.be/g07m6oLJqEk?feature=shared&t=6
