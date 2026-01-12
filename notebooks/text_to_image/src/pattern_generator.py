"""Pattern generation module for text-to-image generation."""

import numpy as np
from typing import List

PATTERN_TYPES: List[str] = [
    "stripe",
    "checkerboard",
    "gradient",
    "dots",
    "wave",
    "spiral",
    "grid",
    "noise"
]


class PatternGenerator:
    """Procedural pattern generator for grayscale images."""

    def __init__(self, image_size: int = 200, seed: int = 42):
        self.image_size = image_size
        self.seed = seed

    def generate(self, pattern_type: str, variation_seed: int = 0) -> np.ndarray:
        """Generate a pattern image with variations.

        Args:
            pattern_type: One of PATTERN_TYPES
            variation_seed: Seed offset for variation in parameters

        Returns:
            2D numpy array of shape (image_size, image_size) with values in [0, 1]
        """
        rng = np.random.default_rng(self.seed + variation_seed)

        generators = {
            "stripe": self._generate_stripe,
            "checkerboard": self._generate_checkerboard,
            "gradient": self._generate_gradient,
            "dots": self._generate_dots,
            "wave": self._generate_wave,
            "spiral": self._generate_spiral,
            "grid": self._generate_grid,
            "noise": self._generate_noise,
        }

        if pattern_type not in generators:
            raise ValueError(f"Unknown pattern: {pattern_type}. Must be one of {PATTERN_TYPES}")

        return generators[pattern_type](rng)

    def _generate_stripe(self, rng: np.random.Generator) -> np.ndarray:
        """Vertical or horizontal stripes with variable width."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        horizontal = rng.random() > 0.5
        stripe_width = rng.integers(8, 35)

        for i in range(0, self.image_size, stripe_width * 2):
            end = min(i + stripe_width, self.image_size)
            if horizontal:
                img[i:end, :] = 1.0
            else:
                img[:, i:end] = 1.0
        return img

    def _generate_checkerboard(self, rng: np.random.Generator) -> np.ndarray:
        """Checkerboard pattern with variable square size."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        square_size = rng.integers(12, 45)

        for i in range(0, self.image_size, square_size):
            for j in range(0, self.image_size, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    i_end = min(i + square_size, self.image_size)
                    j_end = min(j + square_size, self.image_size)
                    img[i:i_end, j:j_end] = 1.0
        return img

    def _generate_gradient(self, rng: np.random.Generator) -> np.ndarray:
        """Linear gradient (horizontal, vertical, or diagonal)."""
        direction = rng.integers(0, 3)  # 0=horizontal, 1=vertical, 2=diagonal

        x = np.linspace(0, 1, self.image_size)
        y = np.linspace(0, 1, self.image_size)
        xx, yy = np.meshgrid(x, y)

        if direction == 0:
            img = xx
        elif direction == 1:
            img = yy
        else:
            img = (xx + yy) / 2

        return img.astype(np.float32)

    def _generate_dots(self, rng: np.random.Generator) -> np.ndarray:
        """Regular dot grid pattern."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        spacing = rng.integers(18, 40)
        radius = rng.integers(4, min(spacing // 3, 12))

        y_coords, x_coords = np.ogrid[:self.image_size, :self.image_size]

        for cy in range(spacing // 2, self.image_size, spacing):
            for cx in range(spacing // 2, self.image_size, spacing):
                mask = (x_coords - cx) ** 2 + (y_coords - cy) ** 2 <= radius ** 2
                img[mask] = 1.0
        return img

    def _generate_wave(self, rng: np.random.Generator) -> np.ndarray:
        """Sinusoidal wave pattern."""
        frequency = rng.uniform(3, 10)
        amplitude = rng.uniform(0.3, 0.45)
        phase = rng.uniform(0, 2 * np.pi)
        horizontal = rng.random() > 0.5

        coords = np.linspace(0, 2 * np.pi * frequency, self.image_size)
        wave = 0.5 + amplitude * np.sin(coords + phase)

        if horizontal:
            img = np.tile(wave[np.newaxis, :], (self.image_size, 1))
        else:
            img = np.tile(wave[:, np.newaxis], (1, self.image_size))

        return img.astype(np.float32)

    def _generate_spiral(self, rng: np.random.Generator) -> np.ndarray:
        """Archimedean spiral pattern."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        center = self.image_size // 2
        arm_count = rng.integers(2, 6)
        thickness = rng.integers(3, 7)

        y_coords, x_coords = np.ogrid[:self.image_size, :self.image_size]
        x_centered = x_coords - center
        y_centered = y_coords - center

        r = np.sqrt(x_centered ** 2 + y_centered ** 2)
        theta = np.arctan2(y_centered, x_centered)

        spiral = (r + theta * arm_count * 3) % 20
        img = (spiral < thickness).astype(np.float32)

        return img

    def _generate_grid(self, rng: np.random.Generator) -> np.ndarray:
        """Cross-hatch grid pattern."""
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        spacing = rng.integers(18, 45)
        line_width = rng.integers(2, 5)

        for i in range(0, self.image_size, spacing):
            end = min(i + line_width, self.image_size)
            img[i:end, :] = 1.0
            img[:, i:end] = 1.0

        return img

    def _generate_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Structured random noise (smooth interpolated)."""
        scale = rng.integers(8, 25)
        small_size = self.image_size // scale + 1
        small = rng.random((small_size, small_size)).astype(np.float32)

        # Simple bilinear upscaling without scipy dependency
        img = self._bilinear_upsample(small, self.image_size)

        return img

    def _bilinear_upsample(self, small: np.ndarray, target_size: int) -> np.ndarray:
        """Bilinear upsampling of a small array to target size."""
        h_small, w_small = small.shape

        # Create output coordinates
        y_out = np.linspace(0, h_small - 1, target_size)
        x_out = np.linspace(0, w_small - 1, target_size)

        # Get integer and fractional parts
        y0 = np.floor(y_out).astype(int)
        x0 = np.floor(x_out).astype(int)
        y1 = np.minimum(y0 + 1, h_small - 1)
        x1 = np.minimum(x0 + 1, w_small - 1)

        fy = (y_out - y0).reshape(-1, 1)
        fx = (x_out - x0).reshape(1, -1)

        # Bilinear interpolation
        result = (
            small[y0][:, x0] * (1 - fy) * (1 - fx) +
            small[y0][:, x1] * (1 - fy) * fx +
            small[y1][:, x0] * fy * (1 - fx) +
            small[y1][:, x1] * fy * fx
        )

        return result.astype(np.float32)

    def generate_dataset(self, samples_per_pattern: int = 500) -> tuple:
        """Generate a complete dataset with all pattern types.

        Args:
            samples_per_pattern: Number of variations per pattern type

        Returns:
            Tuple of (images, labels, pattern_names)
            - images: (N, image_size*image_size) flattened images
            - labels: (N,) integer labels
            - pattern_names: List of pattern type names for each sample
        """
        images = []
        labels = []
        pattern_names = []

        for pattern_idx, pattern_type in enumerate(PATTERN_TYPES):
            for variation in range(samples_per_pattern):
                img = self.generate(pattern_type, variation_seed=variation)
                images.append(img.flatten())
                labels.append(pattern_idx)
                pattern_names.append(pattern_type)

        return (
            np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            pattern_names
        )
