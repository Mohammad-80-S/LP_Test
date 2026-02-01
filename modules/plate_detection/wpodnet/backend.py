from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import _get_perspective_coeffs, to_tensor

from .model import WPODNet


@dataclass(frozen=True)
class Prediction:
    """Prediction result from WPODNet."""
    bounds: List[Tuple[int, int]]
    confidence: float

    def __post_init__(self):
        if len(self.bounds) != 4:
            raise ValueError(f"Expected 4 points, got {len(self.bounds)}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def annotate(
        self,
        canvas: Image.Image,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: int = 1,
    ) -> None:
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(self.bounds, fill=fill, outline=outline, width=width)

    def get_plate_dimensions(self) -> Tuple[int, int]:
        """
        Calculate the actual plate dimensions based on the 4 corner points.
        
        Returns:
            Tuple of (width, height) based on the detected corners
        """
        # bounds order: [top-left, top-right, bottom-right, bottom-left]
        # or similar quadrilateral order
        pts = np.array(self.bounds, dtype=np.float32)
        
        # Calculate width as average of top and bottom edges
        top_width = np.linalg.norm(pts[1] - pts[0])      # top-left to top-right
        bottom_width = np.linalg.norm(pts[2] - pts[3])   # bottom-right to bottom-left
        width = int((top_width + bottom_width) / 2)
        
        # Calculate height as average of left and right edges
        left_height = np.linalg.norm(pts[3] - pts[0])    # top-left to bottom-left
        right_height = np.linalg.norm(pts[2] - pts[1])   # top-right to bottom-right
        height = int((left_height + right_height) / 2)
        
        return max(1, width), max(1, height)

    def warp(
        self, 
        canvas: Image.Image, 
        width: Optional[int] = None, 
        height: Optional[int] = None,
        use_auto_size: bool = True,
        min_width: int = 20,
        min_height: int = 10,
        max_width: int = 400,
        max_height: int = 150
    ) -> Image.Image:
        """
        Warps the image with perspective based on the bounding polygon.

        Args:
            canvas: The image to be warped.
            width: Output width (if None and use_auto_size=True, calculated from corners)
            height: Output height (if None and use_auto_size=True, calculated from corners)
            use_auto_size: If True, calculate size from corner points
            min_width: Minimum output width
            min_height: Minimum output height
            max_width: Maximum output width
            max_height: Maximum output height

        Returns:
            The warped image.
        """
        if use_auto_size and (width is None or height is None):
            auto_width, auto_height = self.get_plate_dimensions()
            width = width if width is not None else auto_width
            height = height if height is not None else auto_height
        
        # Apply constraints
        width = max(min_width, min(max_width, width))
        height = max(min_height, min(max_height, height))
        
        coeffs = _get_perspective_coeffs(
            startpoints=self.bounds,
            endpoints=[
                (0, 0),
                (width, 0),
                (width, height),
                (0, height),
            ],
        )
        return canvas.transform((width, height), Image.Transform.PERSPECTIVE, coeffs)


Q = np.array([
    [-0.5, 0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0, 1.0],
])


class Predictor:
    """Predictor wrapper for WPODNet."""

    def __init__(self, wpodnet: WPODNet):
        self.wpodnet = wpodnet
        self.wpodnet.eval()

    def _resize_to_fixed_ratio(
        self, image: Image.Image, dim_min: int, dim_max: int
    ) -> Image.Image:
        h, w = image.height, image.width
        wh_ratio = max(h, w) / min(h, w)
        side = int(wh_ratio * dim_min)
        bound_dim = min(side + side % self.wpodnet.stride, dim_max)
        factor = bound_dim / max(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)
        
        reg_w_mod = reg_w % self.wpodnet.stride
        if reg_w_mod > 0:
            reg_w += self.wpodnet.stride - reg_w_mod
        reg_h_mod = reg_h % self.wpodnet.stride
        if reg_h_mod > 0:
            reg_h += self.wpodnet.stride - reg_h_mod
        
        return image.resize((reg_w, reg_h))

    def _to_torch_image(self, image: Image.Image) -> torch.Tensor:
        tensor = to_tensor(image)
        return tensor.unsqueeze_(0)

    def _inference(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            probs, affines = self.wpodnet.forward(image)
        probs = np.squeeze(probs.cpu().numpy())[0]
        affines = np.squeeze(affines.cpu().numpy())
        return probs, affines

    def _get_max_anchor(self, probs: np.ndarray) -> Tuple[int, int]:
        return np.unravel_index(probs.argmax(), probs.shape)

    def _get_bounds(
        self,
        affines: np.ndarray,
        anchor_y: int,
        anchor_x: int,
        scaling_ratio: float = 1.0,
    ) -> np.ndarray:
        theta = affines[:, anchor_y, anchor_x].reshape((2, 3))
        theta[0, 0] = max(theta[0, 0], 0.0)
        theta[1, 1] = max(theta[1, 1], 0.0)
        bounds = np.matmul(theta, Q) * self.wpodnet.scale_factor * scaling_ratio
        _, grid_h, grid_w = affines.shape
        bounds[0] = (bounds[0] + anchor_x + 0.5) / grid_w
        bounds[1] = (bounds[1] + anchor_y + 0.5) / grid_h
        return np.transpose(bounds)

    def predict(
        self,
        image: Image.Image,
        scaling_ratio: float = 1.0,
        dim_min: int = 512,
        dim_max: int = 768,
        confidence_threshold: float = 0.9,
    ) -> Optional[Prediction]:
        orig_h, orig_w = image.height, image.width
        resized = self._resize_to_fixed_ratio(image, dim_min=dim_min, dim_max=dim_max)
        resized = self._to_torch_image(resized)
        resized = resized.to(self.wpodnet.device)

        probs, affines = self._inference(resized)
        max_prob = np.amax(probs)

        if max_prob < confidence_threshold:
            return None

        anchor_y, anchor_x = self._get_max_anchor(probs)
        bounds = self._get_bounds(affines, anchor_y, anchor_x, scaling_ratio)
        bounds[:, 0] *= orig_w
        bounds[:, 1] *= orig_h

        return Prediction(
            bounds=[(x, y) for x, y in np.int32(bounds).tolist()],
            confidence=max_prob.item(),
        )