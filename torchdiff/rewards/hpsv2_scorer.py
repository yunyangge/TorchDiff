"""
Human Preference Score (HPS) Reward
Based on HPSv2 library (supports v2.0 and v2.1).
HPSv3 code is not yet publicly available; this implementation uses HPSv2.1
which is currently the best available HPS model.

When HPSv3 code is released, this file can be updated to support it.

Input: list of images (PIL or numpy) + prompts
Output: list of float rewards
"""

import os
import numpy as np
import torch
from typing import List, Union
from PIL import Image


class HPSScorer:
    def __init__(self, device: str = "cuda", hps_version: str = "v2.1"):
        """
        Human Preference Score reward.
        
        :param device: Device to run the model on
        :param hps_version: HPS version to use ('v2.0', 'v2.1')
                           Note: HPSv3 will be supported once its code is released.
        """
        self.device = device
        self.hps_version = hps_version
        
        try:
            import hpsv2
            self.hpsv2 = hpsv2
        except ImportError:
            raise ImportError(
                "hpsv2 package is required. Install with: pip install hpsv2\n"
                "See: https://github.com/tgxs002/HPSv2"
            )

    def _ensure_pil_images(
        self, images: Union[List[Image.Image], List[np.ndarray], np.ndarray]
    ) -> List[Image.Image]:
        """Convert various image formats to PIL Images."""
        pil_images = []
        if isinstance(images, np.ndarray):
            if images.ndim == 4:  # (B, H, W, C) or (B, F, H, W, C)
                for i in range(images.shape[0]):
                    pil_images.append(Image.fromarray(images[i].astype(np.uint8)))
            elif images.ndim == 5:  # (B, F, H, W, C) - take middle frame
                for i in range(images.shape[0]):
                    mid_idx = images.shape[1] // 2
                    pil_images.append(Image.fromarray(images[i, mid_idx].astype(np.uint8)))
        else:
            for img in images:
                if isinstance(img, Image.Image):
                    pil_images.append(img)
                elif isinstance(img, np.ndarray):
                    if img.ndim == 3:  # (H, W, C)
                        pil_images.append(Image.fromarray(img.astype(np.uint8)))
                    elif img.ndim == 4:  # (F, H, W, C) - take middle frame
                        mid_idx = img.shape[0] // 2
                        pil_images.append(Image.fromarray(img[mid_idx].astype(np.uint8)))
                elif isinstance(img, torch.Tensor):
                    if img.ndim == 3:  # (C, H, W)
                        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(arr))
                    elif img.ndim == 4:  # (F, C, H, W) - take middle frame
                        mid_idx = img.shape[0] // 2
                        arr = (img[mid_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        pil_images.append(Image.fromarray(arr))
        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray], np.ndarray],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate HPS reward.
        
        :param images: List of images (PIL, numpy HWC, or numpy FHWC for video)
                       For video inputs, the middle frame is used for scoring.
        :param prompts: List of text prompts
        :return: List of reward scores
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        pil_images = self._ensure_pil_images(images)
        rewards = []
        
        for img, prompt in zip(pil_images, prompts):
            # hpsv2.score accepts: (images, prompt, hps_version)
            # images can be a single PIL image or list of PIL images
            score = self.hpsv2.score([img], prompt, hps_version=self.hps_version)
            # score is a numpy array or list, extract the scalar
            if isinstance(score, (list, np.ndarray)):
                score = float(score[0])
            else:
                score = float(score)
            rewards.append(score)
        
        return rewards


class HPSScorer_video_or_image:
    def __init__(self, device: str = "cuda", hps_version: str = "v2.1"):
        """
        HPS reward calculator for both images and videos.
        For videos, scores multiple sampled frames and averages.
        
        :param device: Device to run the model on
        :param hps_version: HPS version ('v2.0', 'v2.1')
        """
        self.device = device
        self.hps_version = hps_version
        self.frame_interval = 4  # Sample every 4th frame for videos
        
        try:
            import hpsv2
            self.hpsv2 = hpsv2
        except ImportError:
            raise ImportError(
                "hpsv2 package is required. Install with: pip install hpsv2\n"
                "See: https://github.com/tgxs002/HPSv2"
            )

    @torch.no_grad()
    def __call__(
        self,
        images: Union[List[Image.Image], List[np.ndarray]],
        prompts: List[str],
    ) -> List[float]:
        """
        Calculate HPS reward for images or videos.
        
        :param images: List of images or videos
                       - PIL.Image: single image
                       - np.ndarray (H,W,C): single image
                       - np.ndarray (F,H,W,C): video
        :param prompts: List of text prompts
        :return: List of reward scores
        """
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        
        rewards = []
        for img, prompt in zip(images, prompts):
            frame_scores = []
            
            # Handle video: shape (F, H, W, C)
            if isinstance(img, np.ndarray) and img.ndim == 4:
                sampled_frames = img[::self.frame_interval]
                for frame in sampled_frames:
                    pil_frame = Image.fromarray(frame.astype(np.uint8))
                    score = self.hpsv2.score([pil_frame], prompt, hps_version=self.hps_version)
                    score = float(score[0]) if isinstance(score, (list, np.ndarray)) else float(score)
                    frame_scores.append(score)
            else:
                # Single image
                if isinstance(img, Image.Image):
                    pil_img = img
                elif isinstance(img, np.ndarray):
                    pil_img = Image.fromarray(img.astype(np.uint8))
                elif isinstance(img, torch.Tensor):
                    arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    pil_img = Image.fromarray(arr)
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
                
                score = self.hpsv2.score([pil_img], prompt, hps_version=self.hps_version)
                score = float(score[0]) if isinstance(score, (list, np.ndarray)) else float(score)
                frame_scores.append(score)
            
            if frame_scores:
                rewards.append(sum(frame_scores) / len(frame_scores))
            else:
                rewards.append(0.0)
        
        return rewards


if __name__ == "__main__":
    scorer = HPSScorer(device="cuda", hps_version="v2.1")
    test_image = Image.new("RGB", (512, 512), (128, 128, 200))
    reward = scorer([test_image], ["a beautiful mountain landscape at sunset"])
    print(f"HPS Reward: {reward}")
