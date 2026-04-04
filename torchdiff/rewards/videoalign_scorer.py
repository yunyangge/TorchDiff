"""
VideoAlign Reward Scorer

Wraps the VideoAlign model (VideoVLMRewardInference) to work with the
TorchDiff reward interface. VideoAlign provides three evaluation dimensions:
  - VQ (Visual Quality)
  - MQ (Motion Quality)
  - TA (Text Alignment)
  - Overall = VQ + MQ + TA

Since VideoAlign expects video file paths as input, this scorer converts
numpy/tensor video frames into temporary mp4 files before scoring.

All VideoAlign dependencies are now local within torchdiff/rewards/,
no sys.path manipulation is needed.

Input:  numpy array of shape (B, F, H, W, C) uint8 [0,255] + prompts
Output: list of float rewards (Overall score, normalized)
"""

import os
import tempfile
import shutil
import numpy as np
import torch
import torch_npu
from typing import List, Union, Optional
from PIL import Image


class VideoAlignScorer:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_from_pretrained: str = None,
        load_from_pretrained_step: int = -1,
        reward_dim: str = "Overall",  # "VQ", "MQ", "TA", "Overall"
        use_norm: bool = True,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        max_pixels: Optional[int] = None,
        save_fps: int = 8,  # fps for writing temp video files
    ):
        """
        VideoAlign reward scorer.

        :param device: Device to run the model on
        :param dtype: Data type for the model
        :param load_from_pretrained: Path to VideoAlign checkpoint directory
        :param load_from_pretrained_step: Checkpoint step to load (-1 for latest)
        :param reward_dim: Which dimension to use as reward ("VQ", "MQ", "TA", "Overall")
        :param use_norm: Whether to normalize the reward scores
        :param fps: FPS for VideoAlign video sampling (None to use model default)
        :param num_frames: Number of frames for VideoAlign sampling (None to use model default)
        :param max_pixels: Max pixels for VideoAlign (None to use model default)
        :param save_fps: FPS used when saving temp video files from numpy arrays
        """
        self.device = device
        self.reward_dim = reward_dim
        self.use_norm = use_norm
        self.fps = fps
        self.num_frames = num_frames
        self.max_pixels = max_pixels
        self.save_fps = save_fps

        # Import from local module (no sys.path hack needed)
        from torchdiff.rewards.videoalign_inference import VideoVLMRewardInference

        self.inferencer = VideoVLMRewardInference(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            device=device,
            dtype=dtype,
        )

        print(f"[VideoAlignScorer] Initialized with checkpoint: {load_from_pretrained}")
        print(f"[VideoAlignScorer] reward_dim={reward_dim}, use_norm={use_norm}")

    def _save_video_to_tempfile(self, frames: np.ndarray, tmp_dir: str, idx: int) -> str:
        """
        Save numpy video frames to a temporary mp4 file.

        :param frames: (F, H, W, C) uint8 numpy array
        :param tmp_dir: Temporary directory to save to
        :param idx: Index for unique naming
        :return: Path to the saved video file
        """
        from torchvision.io import write_video

        video_path = os.path.join(tmp_dir, f"tmp_video_{idx}.mp4")
        # frames: (F, H, W, C) uint8 -> torch tensor
        video_tensor = torch.from_numpy(frames).to(torch.uint8)
        write_video(video_path, video_tensor, self.save_fps, video_codec="h264")
        return video_path

    def _convert_to_video_frames(self, images) -> List[np.ndarray]:
        """
        Convert various input formats to list of (F, H, W, C) uint8 numpy arrays.

        Supports:
        - np.ndarray (B, F, H, W, C) or (B, H, W, C) 
        - torch.Tensor various layouts
        - List of np.ndarray / PIL.Image
        """
        result = []
        if isinstance(images, np.ndarray):
            if images.ndim == 5:
                # (B, F, H, W, C)
                for i in range(images.shape[0]):
                    result.append(images[i])
            elif images.ndim == 4:
                # (B, H, W, C) - single frame per sample
                for i in range(images.shape[0]):
                    result.append(images[i:i+1])  # (1, H, W, C)
        elif isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy() if images.is_cuda else images.numpy()
            return self._convert_to_video_frames(images_np)
        elif isinstance(images, (list, tuple)):
            for img in images:
                if isinstance(img, np.ndarray):
                    if img.ndim == 4:
                        result.append(img)  # (F, H, W, C)
                    elif img.ndim == 3:
                        result.append(img[np.newaxis])  # (1, H, W, C)
                elif isinstance(img, Image.Image):
                    arr = np.array(img)
                    result.append(arr[np.newaxis])
                elif isinstance(img, torch.Tensor):
                    arr = img.cpu().numpy() if img.is_cuda else img.numpy()
                    if arr.ndim == 4:
                        result.append(arr)
                    elif arr.ndim == 3:
                        result.append(arr[np.newaxis])
        return result

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List, np.ndarray, torch.Tensor],
        prompts: List[str],
    ) -> List[float]:
        """
        Compute VideoAlign reward scores.

        :param images: Video data. Expects (B, F, H, W, C) uint8 [0,255] numpy.
                       Also supports other common formats (auto-converted).
        :param prompts: List of B text prompts.
        :return: List of B reward scores.
        """
        video_frames_list = self._convert_to_video_frames(images)
        assert len(video_frames_list) == len(prompts), \
            f"Number of videos ({len(video_frames_list)}) must match prompts ({len(prompts)})"

        # Create a temporary directory for video files
        tmp_dir = tempfile.mkdtemp(prefix="videoalign_")
        try:
            # Save all videos to temp files
            video_paths = []
            for i, frames in enumerate(video_frames_list):
                if frames.dtype != np.uint8:
                    frames = np.clip(frames * 255 if frames.max() <= 1.0 else frames, 0, 255).astype(np.uint8)
                path = self._save_video_to_tempfile(frames, tmp_dir, i)
                video_paths.append(path)

            # Call VideoAlign inference
            rewards = self.inferencer.reward(
                video_paths=video_paths,
                prompts=prompts,
                fps=self.fps,
                num_frames=self.num_frames,
                max_pixels=self.max_pixels,
                use_norm=self.use_norm,
            )

            # Extract the requested dimension
            scores = [r[self.reward_dim] for r in rewards]
            return scores

        finally:
            # Clean up temp files
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Example usage
    scorer = VideoAlignScorer(
        device="npu",
        load_from_pretrained="/home/ma-user/work/xianyi/ckpts/KlingTeam/VideoReward",
        reward_dim="Overall",
    )

    # Simulate a batch of 2 videos, each with 16 frames at 256x256
    batch_videos = np.random.randint(0, 255, (2, 16, 256, 256, 3), dtype=np.uint8)
    prompts = [
        "A person walking in a park",
        "A cat playing with a ball",
    ]

    scores = scorer(batch_videos, prompts)
    print(f"VideoAlign scores: {scores}")
