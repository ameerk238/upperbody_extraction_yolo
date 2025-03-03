# Upper Body Extraction Tool using YOLO

A tool I developed to extract upper body frames from videos using YOLO pose estimation.

## Research Purpose

I'm conducting research on generating upper body movement using diffusion models. This tool helps me create a high-quality dataset for training various state-of-the-art animation models such as:

- [MooreAnimate](https://github.com/snap-research/articulated-animation)
- [MagicDance](https://boese0601.github.io/magicdance/)
- [LIA (Live Image Animation)](https://wyhsirius.github.io/LIA-project/)
- [MimicMotion](https://tencent.github.io/MimicMotion/)
- [Articulated Animation](https://snap-research.github.io/articulated-animation/)

The goal is to create a robust dataset of human upper body movements that can be used to train models that generate realistic human animations from still images.

## How It Works

I use YOLO's pose estimation capabilities to:
1. Detect people in the video frame
2. Identify and validate upper body pose landmarks
3. Keep only the frames where the upper body is properly visible with good proportions

Through experimentation, I found that detections should occupy between 35% and 47% of the frame area with an aspect ratio between 0.75 and 0.95 for optimal upper body visibility. I also check that important landmarks (head, shoulders, arms) are clearly visible with a visibility score above 0.5.

The script works with videos of any resolution, as it uses relative measurements (percentages) rather than absolute pixel values. This makes it versatile for processing videos from different sources and qualities.

## Example Results

I've tested the tool on various videos, including talks from 99U:

Original videos:
- [Video 1: Original Source](https://www.youtube.com/watch?v=ayQeYjSVd3Y&ab_channel=99U)
- [Video 2: Original Source](https://www.youtube.com/watch?v=R75AfAjsTiQ&ab_channel=99U)

### Video Demo

Since GitHub doesn't directly support playing videos in README files, you can check out these animated GIFs showing the processing in action:

![Video 1 GIF](/upperbody_extraction_yolo/GIFs/video1_yolo.gif)
![Video 2 GIF](/upperbody_extraction_yolo/GIFs/video2_yolo.gif)

**Note:** These example GIFs and videos are just short clips demonstrating the processing technique, not the complete processed output of the entire YouTube videos. They're meant to illustrate how the tool works rather than provide the full dataset.

## Installation

```
pip install -r requirements.txt
```

Make sure you have ffmpeg installed:
```
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Usage

Process videos by pointing to your input and output folders:

```
python video_processor.py --input /path/to/videos --output /path/to/results
```

### Options

- `--input` or `-i`: Folder containing video files
- `--output` or `-o`: Where to save processed videos
- `--processes` or `-p`: Number of parallel processes (default: auto)

## Creating Your Own Dataset

If you want to create a similar dataset:

1. Collect YouTube video IDs of interest
2. Download videos using youtube-dl or similar tools:
   ```
   youtube-dl -f 'bestvideo[height<=720]' -o '%(id)s.%(ext)s' VIDEO_ID
   ```
3. Run this tool to extract good quality upper body frames
4. Use the processed videos for your machine learning projects

## Requirements

- Python 3.6+
- ffmpeg-python
- numpy
- ultralytics (YOLO)
- opencv-python
- psutil
- torch

## Comparison with MediaPipe Version

This tool is a companion to my [earlier MediaPipe-based extraction tool](https://github.com/yourusername/upperbody_extraction_mediapipe). I'm experimenting with both approaches to understand how they work for upper body extraction. The main differences are:

1. Uses YOLO instead of MediaPipe for detection and pose estimation
2. Different parameter thresholds for body proportions
3. Alternative validation criteria for extractions
4. Similar parallel processing capabilities

Both tools serve the same research purpose of creating high-quality datasets for animation models.