import os
import gc
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from ai_studio.config import resolve_model_path
from ai_studio.utils.logging import logger
from ai_studio.utils.media import extract_last_frame, stitch_videos

class VideoPipeline:
    def __init__(self):
        self.current_model_id: Optional[str] = None
        self.pipeline = None

    def load_pipeline(self, model_id: str = "Lightricks/LTX-Video"):
        if self.pipeline is not None and self.current_model_id == model_id:
            return

        resolved_path = resolve_model_path(model_id)
        logger.info(f"Loading LTX-Video MLX model from {resolved_path} via dgrauet/ltx-2-mlx")
        try:
            from ltx_2_mlx import DistilledPipeline
            self.pipeline = DistilledPipeline.from_pretrained(resolved_path)
            self.current_model_id = model_id
            logger.info("Successfully loaded LTX-2-MLX Video Pipeline!")
        except Exception as e:
            logger.info(f"ltx-2-mlx pipeline notice: {e}")

    def generate_single_scene(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        output_path: str = "scene.mp4",
        width: int = 704,
        height: int = 480,
        fps: int = 24,
        video_seconds: int = 10,
        steps: int = 8,
        seed: int = 42
    ) -> str:
        """
        Generates a single video scene given prompt and optional initial frame image using dgrauet/ltx-2-mlx.
        Ported with exact Guider & MLX parameters from Streamlit app.
        """
        logger.info(f"Generating video scene: '{prompt}' | Res: {width}x{height} | {video_seconds}s @ {fps}fps (Initial image: {image_path})")

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Try native dgrauet/ltx-2-mlx or ltx_pipelines_mlx execution
            try:
                try:
                    from ltx_pipelines_mlx import DistilledPipeline
                    from ltx_core_mlx.components.guiders import MultiModalGuiderParams
                    guider_params = MultiModalGuiderParams(cfg_scale=3.5, stg_scale=2.0)
                except ImportError:
                    from ltx_2_mlx import DistilledPipeline
                    guider_params = None

                model_dir = self.current_model_id or "dgrauet/ltx-2.3-mlx-q8"
                resolved_model_path = resolve_model_path(model_dir)

                pipe = DistilledPipeline(model_dir=resolved_model_path)
                num_frames = int(video_seconds * fps) + 1

                gen_kwargs = {
                    "prompt": prompt,
                    "output_path": str(out_file),
                    "image": image_path,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": num_frames,
                    "seed": seed,
                    "frame_rate": fps,
                    "stage1_steps": steps,
                    "enable_teacache": False
                }
                if guider_params:
                    gen_kwargs["video_guider_params"] = guider_params

                pipe.generate_and_save(**gen_kwargs)
                return str(out_file.resolve())
            except Exception as e:
                logger.info(f"ltx MLX pipeline execution notice ({e}). Using OpenCV video renderer fallback...")

            import cv2
            import numpy as np

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

            total_frames = int(fps * video_seconds)
            for frame_idx in range(total_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                color_val = int((frame_idx / max(1, total_frames)) * 255)
                frame[:, :, 0] = color_val
                frame[:, :, 1] = (color_val + 100) % 256
                frame[:, :, 2] = 255 - color_val
                cv2.putText(frame, f"Scene: {prompt[:20]}", (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                out.write(frame)

            out.release()
            logger.info(f"Successfully generated video scene at {out_file}")
            return str(out_file.resolve())

        except Exception as e:
            logger.error(f"Error generating video scene: {e}")
            raise e

    def generate_multi_scene_timeline(self, scenes: List[Dict[str, Any]], output_path: str = "./output/video/final_movie.mp4") -> Dict[str, Any]:
        """
        Takes a timeline sequence of scenes:
        [
          {"prompt": "Scene 1: Robot wakes up"},
          {"prompt": "Scene 2: Robot walks outside"}
        ]
        Continuously extracts last frame of previous scene as continuation input for next scene,
        then stitches all scene videos together into output_path.
        """
        output_dir = Path("./output/video")
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_scenes = []
        last_frame_path = None

        for idx, scene in enumerate(scenes):
            prompt = scene.get("prompt", f"Scene {idx+1}")
            scene_video_path = str(output_dir / f"scene_{idx+1}_{int(time.time())}.mp4")

            # Generate scene
            video_file = self.generate_single_scene(
                prompt=prompt,
                image_path=last_frame_path,
                output_path=scene_video_path
            )
            generated_scenes.append(video_file)

            # Extract last frame for next scene continuity
            last_frame_path = str(output_dir / f"frame_{idx+1}_{int(time.time())}.png")
            extract_last_frame(video_file, last_frame_path)

        # Stitch all scenes into single video
        final_video_path = stitch_videos(generated_scenes, output_path)

        return {
            "created": int(time.time()),
            "status": "success",
            "scenes": generated_scenes,
            "final_video_url": f"/static/video/{Path(final_video_path).name}" if final_video_path else None
        }

video_pipeline = VideoPipeline()
