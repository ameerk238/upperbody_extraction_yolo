from ultralytics import YOLO
import numpy as np
import subprocess
import json
import cv2
import logging
import os
import torch
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_video_processing.log'),
        logging.StreamHandler()
    ]
)

def get_video_info(video_path):
    """Get video dimensions and frame rate using ffprobe"""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
           '-show_entries', 'stream=width,height,r_frame_rate', 
           '-of', 'json', video_path]
    
    try:
        output = subprocess.check_output(cmd).decode('utf-8')
        info = json.loads(output)
        stream = info['streams'][0]
        
        width = int(stream['width'])
        height = int(stream['height'])
        fps_num, fps_den = map(int, stream['r_frame_rate'].split('/'))
        fps = fps_num / fps_den
        
        return width, height, fps
    except subprocess.CalledProcessError as e:
        logging.error(f"Error getting video info for {video_path}: {str(e)}")
        return None, None, None

def validate_detection(box, keypoints, frame_height, frame_width):
    """Validate if the detected person meets criteria based on manual observations"""
    x1, y1, x2, y2 = map(int, box)
    
    box_width = x2 - x1
    box_height = y2 - y1
    aspect_ratio = box_width / box_height
    
    box_area = box_width * box_height
    frame_area = frame_height * frame_width
    area_percentage = (box_area / frame_area) * 100
    
    ratio_valid = 0.75 <= aspect_ratio <= 0.95
    area_valid = 35 <= area_percentage <= 47

    if keypoints is None or len(keypoints) < 17:
        return False, None

    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    wrist_distance = abs(left_wrist[0] - right_wrist[0]) / frame_width
    hands_valid = wrist_distance <= 0.28

    upper_body_points = [0, 5, 6, 7, 8, 9, 10]
    min_conf = min(keypoints[i][2] for i in upper_body_points)
    confidence_valid = min_conf > 0.5

    metrics = {
        'aspect_ratio': aspect_ratio,
        'area_percentage': area_percentage,
        'wrist_distance': wrist_distance,
        'min_conf': min_conf
    }

    is_valid = ratio_valid and area_valid and hands_valid and confidence_valid

    if is_valid:
        logging.info(f"Valid detection - AR: {aspect_ratio:.2f}, Area%: {area_percentage:.1f}, "
                    f"Wrist Dist: {wrist_distance:.2f}, Min Conf: {min_conf:.2f}")

    return is_valid, metrics

def draw_metrics(frame, metrics, x1, y1, x2, y2, frame_width):
    """Draw metrics on the side of the detection box"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_color = (0, 255, 0)
    thickness = 1
    line_spacing = 15

    metrics_text = [
        f"AR: {metrics['aspect_ratio']:.2f}",
        f"Area%: {metrics['area_percentage']:.1f}",
        f"Wrist Dist: {metrics['wrist_distance']:.2f}",
        f"Min Conf: {metrics['min_conf']:.2f}"
    ]

    max_text_width = max([cv2.getTextSize(text, font, font_scale, thickness)[0][0] 
                         for text in metrics_text])

    text_x = x2 + 5 if x2 + max_text_width + 10 < frame_width else x1 - max_text_width - 5

    start_y = y1 + 15
    for text in metrics_text:
        cv2.putText(frame, text, (text_x, start_y), font, font_scale, font_color, thickness)
        start_y += line_spacing

def process_single_video(video_path, output_path, model):
    """Process a single video and save frames with valid person detections"""
    logging.info(f"Processing video: {video_path}")
    
    # Get video information
    width, height, fps = get_video_info(video_path)
    if width is None:
        logging.error(f"Failed to get video info for {video_path}. Skipping...")
        return False
    
    logging.info(f"Video info: {width}x{height} @ {fps}fps")

    # FFmpeg input process
    input_cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        '-'
    ]
    
    # FFmpeg output process
    output_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'rgb24',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        str(output_path)
    ]

    try:
        input_process = subprocess.Popen(input_cmd, stdout=subprocess.PIPE)
        output_process = subprocess.Popen(output_cmd, stdin=subprocess.PIPE)

        frame_size = width * height * 3
        frame_count = 0
        saved_count = 0

        while True:
            raw_frame = input_process.stdout.read(frame_size)
            if not raw_frame:
                break

            frame = np.frombuffer(raw_frame, dtype=np.uint8)
            frame = frame.reshape((height, width, 3))

            frame_count += 1
            if frame_count % 30 == 0:
                logging.info(f"{video_path}: Processing frame {frame_count}")

            # Run detection
            results = model(frame, classes=[0])
            
            for result in results:
                if len(result.boxes) == 1:
                    box = result.boxes[0].xyxy[0].cpu().numpy()
                    keypoints = result.keypoints[0].data[0].cpu().numpy()
                    confidence = result.boxes[0].conf[0].cpu().numpy()
                    
                    if confidence < 0.5:
                        continue
                        
                    is_valid, metrics = validate_detection(box, keypoints, height, width)
                    
                    if is_valid:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        draw_metrics(frame, metrics, x1, y1, x2, y2, width)
                        output_process.stdin.write(frame.tobytes())
                        saved_count += 1

        logging.info(f"Completed {video_path}: Processed {frame_count} frames, saved {saved_count} frames")
        return True

    except Exception as e:
        logging.error(f"Error processing {video_path}: {str(e)}")
        return False

    finally:
        if 'input_process' in locals() and input_process.stdout:
            input_process.stdout.close()
        if 'output_process' in locals() and output_process.stdin:
            output_process.stdin.close()
        if 'input_process' in locals():
            input_process.wait()
        if 'output_process' in locals():
            output_process.wait()

def process_video_folder(input_folder, output_folder, max_workers=None):
    """
    Process all videos in the input folder and save results to the output folder
    
    Args:
        input_folder (str): Path to folder containing input videos
        output_folder (str): Path to save processed videos
        max_workers (int): Maximum number of concurrent video processes
    """
    start_time = time.time()
    
    # Auto-determine number of workers if not specified
    if max_workers is None:
        max_workers = min(psutil.cpu_count(), 4)  # Default to CPU count but cap at 4
    
    # Convert to Path objects for easier handling
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in input_path.glob('**/*') if f.suffix.lower() in video_extensions]
    
    if not video_files:
        logging.error(f"No video files found in {input_folder}")
        return
    
    logging.info(f"Found {len(video_files)} videos to process using {max_workers} workers")
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    
    # Load YOLO model
    model = YOLO('yolo11m-pose.pt')
    model.to(device)
    
    # Process videos
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_file in video_files:
            # Create output path maintaining relative directory structure
            rel_path = video_file.relative_to(input_path)
            output_file = output_path / rel_path.with_name(f"{rel_path.stem}_processed{rel_path.suffix}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Submit processing task
            future = executor.submit(process_single_video, video_file, output_file, model)
            futures.append((video_file, future))
        
        # Wait for all tasks to complete
        for video_file, future in futures:
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logging.error(f"Failed to process {video_file}: {str(e)}")
                failed += 1
    
    # Log summary
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"\nProcessing Summary:")
    logging.info(f"Total videos: {len(video_files)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info(f"Average time per video: {duration/len(video_files):.2f} seconds")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process videos to extract upper body segments using YOLO')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing video files')
    parser.add_argument('--output', '-o', required=True, help='Output folder to save processed videos')
    parser.add_argument('--processes', '-p', type=int, default=None, 
                      help='Number of parallel processes (default: auto)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_video_folder(args.input, args.output, args.processes)