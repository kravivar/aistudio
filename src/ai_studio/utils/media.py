import os
import subprocess
from typing import List, Optional
from pathlib import Path
from ai_studio.utils.logging import logger

def extract_last_frame(video_path: str, output_image_path: str) -> Optional[str]:
    """
    Extracts the last frame of a video file using OpenCV or FFMPEG fallback.
    Returns path to the output image.
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return None

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(output_image_path, frame)
                cap.release()
                logger.info(f"Extracted last frame using OpenCV to {output_image_path}")
                return output_image_path
        cap.release()
    except Exception as e:
        logger.warning(f"OpenCV frame extraction failed: {e}. Trying FFMPEG fallback...")

    # Fallback to ffmpeg command
    try:
        cmd = [
            "ffmpeg", "-y", "-sseof", "-1", "-i", video_path,
            "-update", "1", "-q:v", "1", output_image_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Extracted last frame using FFMPEG fallback to {output_image_path}")
        return output_image_path
    except Exception as e:
        logger.error(f"FFMPEG frame extraction failed: {e}")
        return None


def stitch_videos(video_paths: List[str], output_path: str) -> Optional[str]:
    """
    Stitches multiple video files into a single video file using ffmpeg concat.
    """
    if not video_paths:
        logger.error("No video paths provided for stitching.")
        return None

    if len(video_paths) == 1:
        return video_paths[0]

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    list_file_path = output_dir / "concat_list.txt"

    try:
        with open(list_file_path, "w") as f:
            for vp in video_paths:
                abs_p = Path(vp).resolve()
                f.write(f"file '{abs_p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file_path), "-c", "copy", output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Successfully stitched {len(video_paths)} videos to {output_path}")

        if list_file_path.exists():
            list_file_path.unlink()

        return output_path
    except Exception as e:
        logger.error(f"Video stitching failed: {e}")
        return None
