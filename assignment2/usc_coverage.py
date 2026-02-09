"""
USC Campus Wireless Coverage Simulation - Webapp-Compatible Engine

This module provides coverage calculation that produces IDENTICAL results to the
webapp (coverageEngine.ts). Key features:
- Mulberry32 PRNG with Box-Muller transform for deterministic shadowing
- 1/8 resolution downscaling for fast computation
- 720-ray visibility map for O(1) line-of-sight lookup
- Clean API for simulated annealing optimization

API for optimization:
    calc = CoverageCalculator("usc_map_buildings_filled.png")
    coverage = calc.calculate_coverage([(x1, y1), (x2, y2), ...])  # Returns 0-100
    valid = calc.is_outdoor(x_meters, y_meters)  # Returns bool
    x, y = calc.get_random_outdoor_location()  # Returns (x_m, y_m)

Performance Notes:
- Pure Python: ~300ms per base station (12 BSs = ~3.5 seconds)
- With Numba: ~30ms per base station (12 BSs = ~360ms)
- Install Numba for faster performance: pip install numba
"""


import math
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

# Try to import Numba for JIT acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create dummy decorators when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# =============================================================================
# CONSTANTS (Match webapp exactly)
# =============================================================================

MAP_WIDTH_M = 640.0
MAP_HEIGHT_M = 430.0
TX_POWER_DBM = -10.0
NOISE_DBM = -101.0
SNR_THRESHOLD_DB = 10.0
SHADOW_STD_DB = 4.0
FREQ_HZ = 2.4e9
COMPUTE_SCALE = 8
NUM_RAYS = 720
SEED = 40

# Precomputed constants
INV_TWO_PI = 1.0 / (2.0 * math.pi)
TWO_PI = 2.0 * math.pi


# =============================================================================
# NUMBA-ACCELERATED FUNCTIONS (used if Numba is available)
# =============================================================================

@jit(nopython=True, cache=True)
def _build_visibility_map_numba(
    bs_px: int, bs_py: int, max_r: int,
    small_w: int, small_h: int, small_mask: np.ndarray,
    ray_cos: np.ndarray, ray_sin: np.ndarray
) -> np.ndarray:
    """Numba-accelerated visibility map building."""
    num_rays = len(ray_cos)
    vis_dist_sq = np.zeros(num_rays, dtype=np.float32)

    for ri in range(num_rays):
        cos_a = ray_cos[ri]
        sin_a = ray_sin[ri]
        max_vis_dist = 0

        for step in range(1, max_r + 1):
            px = int(bs_px + cos_a * step + 0.5)
            py = int(bs_py + sin_a * step + 0.5)

            if px < 0 or px >= small_w or py < 0 or py >= small_h:
                break

            if small_mask[py * small_w + px] == 1:
                break

            max_vis_dist = step

        vis_dist_sq[ri] = max_vis_dist * max_vis_dist

    return vis_dist_sq


@jit(nopython=True, cache=True)
def _compute_coverage_numba(
    min_sx: int, max_sx: int, min_sy: int, max_sy: int,
    small_w: int, small_mask: np.ndarray,
    shadow_map: np.ndarray, cov_mask: np.ndarray,
    bxm: float, bym: float, bspx: int, bspy: int,
    m_per_px_x: float, m_per_px_y: float,
    max_range_sq: float, fspl_const: float,
    tx_power: float, noise: float, snr_threshold: float,
    vis_dist_sq: np.ndarray, num_rays: int
) -> None:
    """Numba-accelerated coverage computation for a single BS."""
    inv_two_pi = 1.0 / (2.0 * 3.141592653589793)
    two_pi = 2.0 * 3.141592653589793

    for sy in range(min_sy, max_sy + 1):
        pt_ym = (sy + 0.5) * m_per_px_y
        dy_m = pt_ym - bym
        dy_m_sq = dy_m * dy_m
        dy_px = sy - bspy
        row_off = sy * small_w

        for sx in range(min_sx, max_sx + 1):
            idx = row_off + sx
            if small_mask[idx] == 1:
                continue

            dx_m = (sx + 0.5) * m_per_px_x - bxm
            dist_sq = dx_m * dx_m + dy_m_sq
            if dist_sq > max_range_sq:
                continue

            distance = math.sqrt(dist_sq)
            if distance < 1.0:
                distance = 1.0

            snr = tx_power - (fspl_const + 20.0 * math.log10(distance)) - shadow_map[idx] - noise
            if snr < snr_threshold:
                continue

            # O(1) LOS lookup
            dx_px = sx - bspx
            pix_dist_sq = dx_px * dx_px + dy_px * dy_px
            angle = math.atan2(dy_px, dx_px)
            if angle < 0:
                angle += two_pi
            ray_idx = int(angle * inv_two_pi * num_rays + 0.5)
            if ray_idx >= num_rays:
                ray_idx = 0

            if pix_dist_sq <= vis_dist_sq[ray_idx]:
                cov_mask[idx] = 1

# =============================================================================
# MULBERRY32 PRNG (Exact port from prng.ts)
# =============================================================================

class Mulberry32:
    """Fast seedable 32-bit PRNG matching JavaScript's Mulberry32."""

    def __init__(self, seed: int):
        # JavaScript's |0 operation - convert to signed 32-bit
        self.state = self._to_int32(seed)

    def next(self) -> float:
        """Generate next random number in [0, 1)."""
        # s = (s + 0x6d2b79f5) | 0;
        self.state = self._to_int32(self.state + 0x6D2B79F5)

        # let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = self._imul(self._unsigned_right_shift(self.state, 15) ^ self.state, 1 | self.state)

        # t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        t = self._to_int32(t + self._imul(self._unsigned_right_shift(t, 7) ^ t, 61 | t)) ^ t

        # return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
        result = self._unsigned_right_shift(self._unsigned_right_shift(t, 14) ^ t, 0)
        return result / 4294967296.0

    @staticmethod
    def _to_int32(x: int) -> int:
        """Convert to signed 32-bit integer (JavaScript's |0 operation)."""
        x = x & 0xFFFFFFFF
        if x >= 0x80000000:
            x -= 0x100000000
        return x

    @staticmethod
    def _unsigned_right_shift(x: int, bits: int) -> int:
        """JavaScript's >>> operator."""
        return (x & 0xFFFFFFFF) >> bits

    @staticmethod
    def _imul(a: int, b: int) -> int:
        """Emulate JavaScript's Math.imul (32-bit integer multiplication)."""
        # Convert to unsigned 32-bit for computation
        a = a & 0xFFFFFFFF
        b = b & 0xFFFFFFFF
        # Low 16 bits of each
        al = a & 0xFFFF
        ah = a >> 16
        bl = b & 0xFFFF
        bh = b >> 16
        # Compute the 32-bit result
        result = (al * bl + (((ah * bl + al * bh) & 0xFFFF) << 16)) & 0xFFFFFFFF
        # Convert to signed 32-bit
        if result >= 0x80000000:
            result -= 0x100000000
        return result


class SeededRNG:
    """Seeded RNG with Box-Muller transform for Gaussian distribution.

    Exact port of prng.ts SeededRNG class.
    """

    def __init__(self, seed: int):
        self.rng = Mulberry32(seed)
        self.spare: Optional[float] = None

    def uniform(self) -> float:
        """Generate uniform random number in [0, 1)."""
        return self.rng.next()

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Generate normally distributed random number using Box-Muller."""
        if self.spare is not None:
            val = self.spare
            self.spare = None
            return mean + std * val

        while True:
            u = 2.0 * self.rng.next() - 1.0
            v = 2.0 * self.rng.next() - 1.0
            s = u * u + v * v
            if s < 1.0 and s != 0.0:
                break

        mul = math.sqrt(-2.0 * math.log(s) / s)
        self.spare = v * mul
        return mean + std * u * mul

    def normal_array(self, length: int, mean: float = 0.0, std: float = 1.0) -> np.ndarray:
        """Generate array of normally distributed random numbers."""
        arr = np.zeros(length, dtype=np.float32)
        for i in range(length):
            arr[i] = self.normal(mean, std)
        return arr


# =============================================================================
# TAB10 COLOR PALETTE (Match webapp exactly)
# =============================================================================

TAB10 = [
    (0.12, 0.47, 0.71),
    (1.0, 0.50, 0.05),
    (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74),
    (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76),
    (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81),
]


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB (matching webapp's hsvToRgb)."""
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return (v, t, p)
    elif i == 1:
        return (q, v, p)
    elif i == 2:
        return (p, v, t)
    elif i == 3:
        return (p, q, v)
    elif i == 4:
        return (t, p, v)
    else:
        return (v, p, q)


def get_bs_colors(count: int) -> List[Tuple[float, float, float]]:
    """Get base station colors (matching webapp's getBsColors)."""
    if count <= 10:
        return list(TAB10[:count])
    return [hsv_to_rgb(i / 20.0 if count <= 20 else i / count, 0.8, 0.9)
            for i in range(count)]


# =============================================================================
# COVERAGE CALCULATOR CLASS
# =============================================================================

class CoverageCalculator:
    """Coverage calculator matching webapp's coverageEngine.ts exactly.

    Usage:
        calc = CoverageCalculator("usc_map_buildings_filled.png")
        coverage = calc.calculate_coverage([(100.0, 200.0), (300.0, 150.0)])
        print(f"Coverage: {coverage:.1f}%")
    """

    def __init__(self, building_mask_path: str):
        """Initialize calculator with building mask image.

        Args:
            building_mask_path: Path to building mask image (black=building, white=outdoor)
        """
        # Load image as grayscale
        img = Image.open(building_mask_path).convert("L")
        self.img_np = np.array(img)
        self.img_height, self.img_width = self.img_np.shape

        # Create binary mask (1=building, 0=outdoor) as Uint8
        # This matches webapp: mask[i] === 1 means building
        self.full_mask = (self.img_np < 128).astype(np.uint8)

        # Precompute downscaled mask
        self._init_downscaled_mask()

        # For get_random_outdoor_location
        self._outdoor_coords_full = None
        self._outdoor_rng = None

    def _init_downscaled_mask(self):
        """Initialize downscaled mask matching webapp's downscaleMask()."""
        full_w = self.img_width
        full_h = self.img_height
        scale = COMPUTE_SCALE

        self.small_w = math.ceil(full_w / scale)
        self.small_h = math.ceil(full_h / scale)
        self.small_total = self.small_w * self.small_h

        self.small_mask = np.zeros(self.small_total, dtype=np.uint8)
        self.small_outdoor_count = 0

        for sy in range(self.small_h):
            fy = min(sy * scale + (scale >> 1), full_h - 1)
            for sx in range(self.small_w):
                fx = min(sx * scale + (scale >> 1), full_w - 1)
                val = self.full_mask[fy, fx]
                idx = sy * self.small_w + sx
                self.small_mask[idx] = val
                if val == 0:  # outdoor
                    self.small_outdoor_count += 1

        # Precompute meters per pixel for small grid
        self.m_per_px_x = MAP_WIDTH_M / self.small_w
        self.m_per_px_y = MAP_HEIGHT_M / self.small_h

    def calculate_coverage(
        self,
        bs_locations: List[Tuple[float, float]],
        tx_power: float = TX_POWER_DBM,
        noise: float = NOISE_DBM,
        snr_threshold: float = SNR_THRESHOLD_DB,
        shadow_std: float = SHADOW_STD_DB,
        freq_hz: float = FREQ_HZ,
        seed: int = SEED
    ) -> float:
        """Calculate coverage percentage for given base station locations.

        This is the main API for optimization. Results match webapp exactly.

        Args:
            bs_locations: List of (x_meters, y_meters) tuples in 640x430m space
            tx_power: Transmit power in dBm (default: -10.0)
            noise: Noise floor in dBm (default: -101.0)
            snr_threshold: Minimum SNR in dB (default: 10.0)
            shadow_std: Shadowing standard deviation in dB (default: 4.0)
            freq_hz: Carrier frequency in Hz (default: 2.4e9)
            seed: Random seed for shadowing (default: 42)

        Returns:
            Coverage percentage (0-100)
        """
        result = self.calculate_coverage_detailed(
            bs_locations, tx_power, noise, snr_threshold, shadow_std, freq_hz, seed
        )
        return result['coverage_percent']

    def calculate_coverage_with_uncovered(
        self,
        bs_locations: List[Tuple[float, float]],
        tx_power: float = TX_POWER_DBM,
        noise: float = NOISE_DBM,
        snr_threshold: float = SNR_THRESHOLD_DB,
        shadow_std: float = SHADOW_STD_DB,
        freq_hz: float = FREQ_HZ,
        seed: int = SEED
    ) -> Tuple[float, np.ndarray]:
        """Calculate coverage percentage and uncovered outdoor pixel mask.

        Returns:
            (coverage_percent, uncovered_mask) where uncovered_mask is a flat
            np.uint8 array of shape (small_total,) with 1 = uncovered outdoor pixel.
        """
        result = self.calculate_coverage_detailed(
            bs_locations, tx_power, noise, snr_threshold, shadow_std, freq_hz, seed
        )
        return result['coverage_percent'], result['uncovered_mask']

    def calculate_coverage_detailed(
        self,
        bs_locations: List[Tuple[float, float]],
        tx_power: float = TX_POWER_DBM,
        noise: float = NOISE_DBM,
        snr_threshold: float = SNR_THRESHOLD_DB,
        shadow_std: float = SHADOW_STD_DB,
        freq_hz: float = FREQ_HZ,
        seed: int = SEED
    ) -> dict:
        """Calculate coverage with full details (for visualization).

        Returns dict with:
            - coverage_percent: float 0-100
            - max_range: float in meters
            - image_data: numpy array (small_h, small_w, 4) RGBA uint8
            - render_width: int
            - render_height: int
            - bs_colors: list of (r, g, b) tuples (0-255)
        """
        num_bs = len(bs_locations)
        max_range = self._calculate_max_range(tx_power, noise, snr_threshold, shadow_std, freq_hz)

        # Build small-res RGBA image (white outdoor, gray buildings)
        image_data = np.zeros((self.small_h, self.small_w, 4), dtype=np.uint8)
        for i in range(self.small_total):
            sy = i // self.small_w
            sx = i % self.small_w
            if self.small_mask[i] == 1:  # building
                image_data[sy, sx] = [77, 77, 77, 255]
            else:  # outdoor
                image_data[sy, sx] = [255, 255, 255, 255]

        if num_bs == 0:
            return {
                'coverage_percent': 0.0,
                'max_range': max_range,
                'image_data': image_data,
                'render_width': self.small_w,
                'render_height': self.small_h,
                'bs_colors': []
            }

        # Precompute constants
        wavelength = 3e8 / freq_hz
        fspl_const = 20.0 * math.log10((4.0 * math.pi) / wavelength)
        max_range_sq = max_range * max_range
        max_range_spx_x = max_range / self.m_per_px_x
        max_range_spx_y = max_range / self.m_per_px_y
        max_range_spx = max(max_range_spx_x, max_range_spx_y)

        # BS positions in small-grid coords and meters
        bs_small_px = np.zeros(num_bs, dtype=np.int32)
        bs_small_py = np.zeros(num_bs, dtype=np.int32)
        bs_xm = np.zeros(num_bs, dtype=np.float64)
        bs_ym = np.zeros(num_bs, dtype=np.float64)

        for i, (x_m, y_m) in enumerate(bs_locations):
            # Convert meters to full-resolution pixels, then to small grid
            full_px = x_m / MAP_WIDTH_M * self.img_width
            full_py = y_m / MAP_HEIGHT_M * self.img_height
            bs_small_px[i] = max(0, min(self.small_w - 1, round(full_px / COMPUTE_SCALE)))
            bs_small_py[i] = max(0, min(self.small_h - 1, round(full_py / COMPUTE_SCALE)))
            bs_xm[i] = x_m
            bs_ym[i] = y_m

        # Shadowing maps (small res) - uses SeededRNG for exact match
        rng = SeededRNG(seed)
        shadowing_maps = []
        for i in range(num_bs):
            shadowing_maps.append(rng.normal_array(self.small_total, 0.0, shadow_std))

        # Coverage masks
        coverage_masks = [np.zeros(self.small_total, dtype=np.uint8) for _ in range(num_bs)]

        # Ensure ray table is ready
        self._ensure_ray_table()

        # Main computation
        for bs_idx in range(num_bs):
            bxm = bs_xm[bs_idx]
            bym = bs_ym[bs_idx]
            bspx = bs_small_px[bs_idx]
            bspy = bs_small_py[bs_idx]

            # Build visibility map for this BS
            vis_dist_sq = self._build_visibility_map(bspx, bspy, max_range_spx)

            # Compute bounds
            min_sx = max(0, int(bspx - max_range_spx_x))
            max_sx = min(self.small_w - 1, math.ceil(bspx + max_range_spx_x))
            min_sy = max(0, int(bspy - max_range_spx_y))
            max_sy = min(self.small_h - 1, math.ceil(bspy + max_range_spx_y))

            shadow_map = shadowing_maps[bs_idx]
            cov_mask = coverage_masks[bs_idx]

            if NUMBA_AVAILABLE:
                # Use Numba-accelerated version
                _compute_coverage_numba(
                    min_sx, max_sx, min_sy, max_sy,
                    self.small_w, self.small_mask,
                    shadow_map, cov_mask,
                    bxm, bym, bspx, bspy,
                    self.m_per_px_x, self.m_per_px_y,
                    max_range_sq, fspl_const,
                    tx_power, noise, snr_threshold,
                    vis_dist_sq, NUM_RAYS
                )
            else:
                # Pure Python with NumPy vectorization
                sy_range = np.arange(min_sy, max_sy + 1)
                sx_range = np.arange(min_sx, max_sx + 1)

                pt_ym = (sy_range + 0.5) * self.m_per_px_y
                pt_xm = (sx_range + 0.5) * self.m_per_px_x

                sx_grid, sy_grid = np.meshgrid(sx_range, sy_range)
                xm_grid, ym_grid = np.meshgrid(pt_xm, pt_ym)

                dx_m = xm_grid - bxm
                dy_m = ym_grid - bym
                dist_sq = dx_m * dx_m + dy_m * dy_m

                dx_px = sx_grid - bspx
                dy_px = sy_grid - bspy
                pix_dist_sq = dx_px * dx_px + dy_px * dy_px

                angles = np.arctan2(dy_px, dx_px)
                angles = np.where(angles < 0, angles + TWO_PI, angles)
                ray_indices = np.clip((angles * INV_TWO_PI * NUM_RAYS + 0.5).astype(np.int32), 0, NUM_RAYS - 1)

                vis_lookup = vis_dist_sq[ray_indices]
                indices = sy_grid * self.small_w + sx_grid
                mask_vals = self.small_mask[indices]
                shadow_vals = shadow_map[indices]

                distance = np.sqrt(dist_sq)
                distance = np.maximum(distance, 1.0)

                snr = tx_power - (fspl_const + 20.0 * np.log10(distance)) - shadow_vals - noise

                valid = (mask_vals == 0) & (dist_sq <= max_range_sq) & (snr >= snr_threshold) & (pix_dist_sq <= vis_lookup)
                cov_mask[indices[valid]] = 1

        # Colors
        colors_float = get_bs_colors(num_bs)
        bs_colors = [(int(c[0] * 255 + 0.5), int(c[1] * 255 + 0.5), int(c[2] * 255 + 0.5))
                     for c in colors_float]

        # Render directly into small-res image
        alpha = 0.35
        one_minus_alpha = 0.65
        total_covered = 0
        uncovered_mask = np.zeros(self.small_total, dtype=np.uint8)

        col_r = np.array([alpha * c[0] for c in colors_float], dtype=np.float32)
        col_g = np.array([alpha * c[1] for c in colors_float], dtype=np.float32)
        col_b = np.array([alpha * c[2] for c in colors_float], dtype=np.float32)

        for i in range(self.small_total):
            if self.small_mask[i] == 1:  # building
                continue

            covered = False
            r, g, b = 1.0, 1.0, 1.0

            for bs_idx in range(num_bs):
                if coverage_masks[bs_idx][i] == 1:
                    covered = True
                    r = col_r[bs_idx] + one_minus_alpha * r
                    g = col_g[bs_idx] + one_minus_alpha * g
                    b = col_b[bs_idx] + one_minus_alpha * b

            if covered:
                total_covered += 1
                sy = i // self.small_w
                sx = i % self.small_w
                image_data[sy, sx, 0] = int(r * 255 + 0.5)
                image_data[sy, sx, 1] = int(g * 255 + 0.5)
                image_data[sy, sx, 2] = int(b * 255 + 0.5)
            else:
                uncovered_mask[i] = 1

        coverage_percent = (100.0 * total_covered / self.small_outdoor_count) if self.small_outdoor_count > 0 else 0.0

        return {
            'coverage_percent': coverage_percent,
            'max_range': max_range,
            'image_data': image_data,
            'render_width': self.small_w,
            'render_height': self.small_h,
            'bs_colors': bs_colors,
            'uncovered_mask': uncovered_mask
        }

    def _ensure_ray_table(self):
        """Ensure ray lookup table is computed."""
        if not hasattr(self, '_ray_cos') or self._ray_cos is None:
            angles = np.arange(NUM_RAYS, dtype=np.float64) / NUM_RAYS * TWO_PI
            self._ray_cos = np.cos(angles)
            self._ray_sin = np.sin(angles)

    def _build_visibility_map(self, bs_px: int, bs_py: int, max_range_px: float) -> np.ndarray:
        """Build visibility map for a base station (matching webapp's buildVisibilityMap).

        Uses Numba JIT if available, otherwise pure Python.
        """
        max_r = int(math.ceil(max_range_px))
        self._ensure_ray_table()

        if NUMBA_AVAILABLE:
            return _build_visibility_map_numba(
                bs_px, bs_py, max_r,
                self.small_w, self.small_h, self.small_mask,
                self._ray_cos, self._ray_sin
            )

        # Pure Python fallback
        vis_dist_sq = np.zeros(NUM_RAYS, dtype=np.float32)
        small_w = self.small_w
        small_h = self.small_h
        small_mask = self.small_mask

        for ri in range(NUM_RAYS):
            cos_a = self._ray_cos[ri]
            sin_a = self._ray_sin[ri]
            max_vis_dist = 0

            for step in range(1, max_r + 1):
                px = int(bs_px + cos_a * step + 0.5)
                py = int(bs_py + sin_a * step + 0.5)

                if px < 0 or px >= small_w or py < 0 or py >= small_h:
                    break

                if small_mask[py * small_w + px] == 1:
                    break

                max_vis_dist = step

            vis_dist_sq[ri] = max_vis_dist * max_vis_dist

        return vis_dist_sq

    def _calculate_max_range(
        self,
        tx_power: float,
        noise: float,
        snr_threshold: float,
        shadow_std: float,
        freq_hz: float
    ) -> float:
        """Calculate maximum possible coverage range (matching webapp's calculateMaxRange)."""
        c = 3e8
        wavelength = c / freq_hz
        max_path_loss = tx_power - noise - snr_threshold + 3 * shadow_std
        return (wavelength / (4 * math.pi)) * (10 ** (max_path_loss / 20))

    def is_outdoor(self, x_meters: float, y_meters: float) -> bool:
        """Check if a location is outdoor (valid for BS placement).

        Args:
            x_meters: X coordinate in meters (0 to MAP_WIDTH_M)
            y_meters: Y coordinate in meters (0 to MAP_HEIGHT_M)

        Returns:
            True if the location is outdoor, False if it's a building
        """
        # Convert to full-resolution pixel coordinates
        px = int(x_meters / MAP_WIDTH_M * self.img_width)
        py = int(y_meters / MAP_HEIGHT_M * self.img_height)

        # Clamp to valid bounds
        px = max(0, min(self.img_width - 1, px))
        py = max(0, min(self.img_height - 1, py))

        return self.full_mask[py, px] == 0

    def get_random_outdoor_location(self, rng: Optional[SeededRNG] = None) -> Tuple[float, float]:
        """Get a random valid outdoor location for BS placement.

        Args:
            rng: Optional SeededRNG instance for reproducibility.
                 If None, uses internal RNG seeded with 42.

        Returns:
            (x_meters, y_meters) tuple
        """
        # Lazy initialization of outdoor coordinates
        if self._outdoor_coords_full is None:
            outdoor_indices = np.where(self.full_mask.flatten() == 0)[0]
            self._outdoor_coords_full = outdoor_indices
            self._outdoor_rng = SeededRNG(SEED)

        if rng is None:
            rng = self._outdoor_rng

        # Pick random outdoor pixel
        rand_idx = int(rng.uniform() * len(self._outdoor_coords_full))
        pixel_idx = self._outdoor_coords_full[rand_idx]

        py = pixel_idx // self.img_width
        px = pixel_idx % self.img_width

        # Convert to meters
        x_m = (px + 0.5) / self.img_width * MAP_WIDTH_M
        y_m = (py + 0.5) / self.img_height * MAP_HEIGHT_M

        return (x_m, y_m)

    def get_max_range(
        self,
        tx_power: float = TX_POWER_DBM,
        noise: float = NOISE_DBM,
        snr_threshold: float = SNR_THRESHOLD_DB,
        shadow_std: float = SHADOW_STD_DB,
        freq_hz: float = FREQ_HZ
    ) -> float:
        """Get maximum coverage range for given parameters.

        Args:
            tx_power: Transmit power in dBm
            noise: Noise floor in dBm
            snr_threshold: Minimum SNR in dB
            shadow_std: Shadowing standard deviation in dB
            freq_hz: Carrier frequency in Hz

        Returns:
            Maximum range in meters
        """
        return self._calculate_max_range(tx_power, noise, snr_threshold, shadow_std, freq_hz)


# =============================================================================
# SIMULATED ANNEALING OPTIMIZATION
# =============================================================================

def sa_cost_function(
    num_bs: int,
    coverage: float,
    target_coverage: float,
    w_bs: float = 1.0,
    w_coverage: float = 100.0
) -> float:
    """Cost function for simulated annealing optimization.

    Minimizes the number of base stations while penalizing coverage shortfall.
    When coverage >= target, cost is simply the BS count. When coverage < target,
    a quadratic penalty drives the search toward feasible solutions.

    Args:
        num_bs: Number of base stations in current solution.
        coverage: Current coverage percentage (0-100).
        target_coverage: Required coverage percentage (e.g. 90.0 or 99.0).
        w_bs: Weight for the base station count term.
        w_coverage: Weight for the coverage penalty term.

    Returns:
        Scalar cost value (lower is better).
    """
    shortfall = max(0.0, target_coverage - coverage)
    return w_bs * num_bs + w_coverage * shortfall * shortfall


def _pick_diverse_location(
    calc: 'CoverageCalculator',
    existing_locations: List[Tuple[float, float]],
    rng: 'SeededRNG',
    num_candidates: int = 50,
    coverage_percent: float = 0.0,
    uncovered_mask: Optional[np.ndarray] = None
) -> Tuple[float, float]:
    """Pick a new BS location using spatial diversity, with gap-filling at high coverage.

    Below 85% coverage: maximizes minimum distance to existing BSs (spread out).
    At 85%+ coverage (with uncovered_mask): scores candidates by the density of
    uncovered outdoor pixels in their vicinity, placing new BSs where they can
    cover the most remaining gaps.

    Args:
        calc: CoverageCalculator instance.
        existing_locations: Current list of (x_meters, y_meters) BS positions.
        rng: SeededRNG instance for reproducibility.
        num_candidates: Number of random candidates to evaluate (default: 50).
        coverage_percent: Current coverage percentage, used to switch modes.
        uncovered_mask: Flat uint8 array (small_total,) with 1 = uncovered outdoor pixel.
                        Required for gap-filling mode at 85%+.

    Returns:
        (x_meters, y_meters) tuple for the selected candidate.
    """
    if not existing_locations:
        return calc.get_random_outdoor_location(rng)

    gap_fill_mode = coverage_percent >= 85.0 and uncovered_mask is not None
    # Search radius in small-grid pixels (~50m at 1/8 scale)
    search_r = 25

    best_loc = None
    best_score = -1.0

    for _ in range(num_candidates):
        x, y = calc.get_random_outdoor_location(rng)

        if gap_fill_mode:
            # Convert candidate to small-grid coordinates
            sx = int(x / MAP_WIDTH_M * calc.small_w)
            sy = int(y / MAP_HEIGHT_M * calc.small_h)
            sx = max(0, min(calc.small_w - 1, sx))
            sy = max(0, min(calc.small_h - 1, sy))

            # Count uncovered pixels within search radius
            uncovered_count = 0
            r_sq = search_r * search_r
            y_lo = max(0, sy - search_r)
            y_hi = min(calc.small_h - 1, sy + search_r)
            x_lo = max(0, sx - search_r)
            x_hi = min(calc.small_w - 1, sx + search_r)
            for py in range(y_lo, y_hi + 1):
                dy = py - sy
                dy_sq = dy * dy
                row_off = py * calc.small_w
                for px in range(x_lo, x_hi + 1):
                    dx = px - sx
                    if dx * dx + dy_sq <= r_sq:
                        if uncovered_mask[row_off + px] == 1:
                            uncovered_count += 1

            score = float(uncovered_count)
        else:
            # Spread mode: maximize minimum distance to existing BSs
            min_dist = float('inf')
            for ex, ey in existing_locations:
                d = (x - ex) ** 2 + (y - ey) ** 2
                if d < min_dist:
                    min_dist = d
            score = min_dist

        if score > best_score:
            best_score = score
            best_loc = (x, y)

    return best_loc


def sa_neighbor(
    calc: 'CoverageCalculator',
    current_locations: List[Tuple[float, float]],
    rng: 'SeededRNG',
    move_radius: float = 50.0,
    coverage_percent: float = 0.0,
    uncovered_mask: Optional[np.ndarray] = None
) -> List[Tuple[float, float]]:
    """Generate a neighbor solution by perturbing the current BS placement.

    Three move types are chosen at random:
      - Move (50%): Shift a random BS to a nearby outdoor location.
                     At 85%+ coverage, move radius shrinks for fine-tuning.
      - Add  (35%): Insert a new BS. At 85%+ with uncovered mask, targets
                     areas with the most uncovered pixels.
      - Remove (15%): Remove a random BS (only if more than 1 remain).

    All generated locations are guaranteed to be outdoor via calc.is_outdoor().

    Args:
        calc: CoverageCalculator instance (provides is_outdoor and map bounds).
        current_locations: Current list of (x_meters, y_meters) BS positions.
        rng: SeededRNG instance for reproducibility.
        move_radius: Maximum displacement in meters for the move operation.
        coverage_percent: Current coverage percentage, affects fine-tuning behavior.
        uncovered_mask: Flat uint8 array (small_total,) with 1 = uncovered outdoor pixel.

    Returns:
        New list of (x_meters, y_meters) BS positions.
    """
    neighbor = list(current_locations)
    n = len(neighbor)
    fine_tune = coverage_percent >= 85.0

    # In fine-tune mode, use a tighter move radius
    effective_move_radius = move_radius * 0.5 if fine_tune else move_radius

    r = rng.uniform()

    if r < 0.5 and n > 0:
        # Move: perturb a random BS position
        idx = int(rng.uniform() * n) % n
        x_old, y_old = neighbor[idx]
        for _ in range(50):  # retry until outdoor location found
            dx = rng.normal(0.0, effective_move_radius)
            dy = rng.normal(0.0, effective_move_radius)
            x_new = max(0.0, min(MAP_WIDTH_M, x_old + dx))
            y_new = max(0.0, min(MAP_HEIGHT_M, y_old + dy))
            if calc.is_outdoor(x_new, y_new):
                neighbor[idx] = (x_new, y_new)
                break

    elif r < 0.85:
        # Add: insert a new BS (gap-filling at 85%+ via uncovered mask)
        x_new, y_new = _pick_diverse_location(
            calc, neighbor, rng, num_candidates=50,
            coverage_percent=coverage_percent,
            uncovered_mask=uncovered_mask
        )
        neighbor.append((x_new, y_new))

    elif n > 1:
        # Remove: drop a random BS
        idx = int(rng.uniform() * n) % n
        neighbor.pop(idx)

    else:
        # Only 1 BS and remove was selected — move it instead
        x_old, y_old = neighbor[0]
        for _ in range(50):
            dx = rng.normal(0.0, effective_move_radius)
            dy = rng.normal(0.0, effective_move_radius)
            x_new = max(0.0, min(MAP_WIDTH_M, x_old + dx))
            y_new = max(0.0, min(MAP_HEIGHT_M, y_old + dy))
            if calc.is_outdoor(x_new, y_new):
                neighbor[0] = (x_new, y_new)
                break

    return neighbor


def simulated_annealing(
    calc: 'CoverageCalculator',
    target_coverage: float = 95.0,
    # --- SA parameters (aggressive defaults for speed) ---
    T_init: float = 50.0,
    T_min: float = 2.0,
    alpha: float = 0.80,
    max_iter_per_temp: int = 8,
    # --- Cost function weights ---
    w_bs: float = 1.0,
    w_coverage: float = 100.0,
    # --- Neighbor parameters ---
    move_radius: float = 50.0,
    # --- Initial solution ---
    initial_num_bs: int = 8,
    seed: int = SEED,
    verbose: bool = True
) -> dict:
    """Simulated annealing to find the minimum number of outdoor base stations
    that achieves a target coverage percentage.

    The algorithm jointly optimizes BS count and positions. At each step a
    neighbor solution is generated (move / add / remove a BS), evaluated with
    the cost function, and accepted or rejected according to the Metropolis
    criterion.  Uses exact full-grid coverage evaluation (with Numba
    acceleration when available).

    Args:
        calc: CoverageCalculator instance.
        target_coverage: Required coverage percentage (e.g. 90.0 or 99.0).
        T_init: Initial temperature.
        T_min: Temperature at which cooling stops.
        alpha: Geometric cooling factor (T *= alpha each outer loop).
        max_iter_per_temp: Number of neighbor evaluations per temperature step.
        w_bs: Cost weight for the number of base stations.
        w_coverage: Cost weight for coverage shortfall penalty.
        move_radius: Max perturbation in meters for the move neighbor.
        initial_num_bs: Number of BSs in the random initial solution.
        seed: Random seed for reproducibility.
        verbose: If True, print progress during optimization.

    Returns:
        Dictionary with keys:
            - locations: Best list of (x_meters, y_meters) BS positions.
            - num_bs: Number of base stations in the best solution.
            - coverage: Coverage percentage of the best solution.
            - cost: Cost value of the best solution.
            - target_coverage: The target that was requested.
            - history: List of (temperature, best_cost, best_coverage, best_num_bs)
                       recorded at each temperature step.
    """
    import time
    t_start = time.time()
    rng = SeededRNG(seed)

    # --- Build initial solution using spatial diversity ---
    # Place BSs far apart to maximize initial coverage
    current_locations = []
    for _ in range(initial_num_bs):
        # Use high candidate count (100) for excellent initial spread
        loc = _pick_diverse_location(calc, current_locations, rng, num_candidates=100)
        current_locations.append(loc)

    current_coverage, current_uncovered = calc.calculate_coverage_with_uncovered(current_locations)
    current_cost = sa_cost_function(len(current_locations), current_coverage,
                                    target_coverage, w_bs, w_coverage)

    best_locations = list(current_locations)
    best_cost = current_cost
    best_coverage = current_coverage
    best_uncovered = current_uncovered

    history = []
    T = T_init
    max_total_iters = 10000  # Safety limit to prevent infinite loops
    total_iters = 0

    if verbose:
        print(f"\n{'='*60}")
        print(f"Simulated Annealing — Target Coverage: {target_coverage}%")
        print(f"{'='*60}")
        print(f"  T_init={T_init}, T_min={T_min}, alpha={alpha}")
        print(f"  max_iter_per_temp={max_iter_per_temp}, move_radius={move_radius}m")
        print(f"  w_bs={w_bs}, w_coverage={w_coverage}")
        print(f"  Numba: {'enabled' if NUMBA_AVAILABLE else 'disabled (install for 10x speedup)'}")
        print(f"  Initial BSs: {initial_num_bs}, Seed: {seed}")
        print(f"  Mode: Continue until target coverage is reached")
        print(f"  Optimization: Spatial diversity (50 candidates, gap-fill at 85%+)")
        print(f"{'='*60}")
        print(f"{'Temp':>10s} {'Cost':>10s} {'Coverage':>10s} {'#BS':>5s}")
        print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*5}")

    # Continue until target coverage is reached or safety limit is hit
    while best_coverage < target_coverage and total_iters < max_total_iters:
        for _ in range(max_iter_per_temp):
            # Generate neighbor (gap-fills using uncovered mask at 85%+)
            neighbor_locations = sa_neighbor(calc, current_locations, rng, move_radius,
                                             coverage_percent=best_coverage,
                                             uncovered_mask=best_uncovered)

            neighbor_coverage, neighbor_uncovered = calc.calculate_coverage_with_uncovered(neighbor_locations)
            neighbor_cost = sa_cost_function(len(neighbor_locations),
                                             neighbor_coverage,
                                             target_coverage, w_bs, w_coverage)

            delta = neighbor_cost - current_cost

            # Metropolis acceptance criterion
            if delta < 0:
                accept = True
            else:
                accept = rng.uniform() < math.exp(-delta / T)

            if accept:
                current_locations = neighbor_locations
                current_cost = neighbor_cost
                current_coverage = neighbor_coverage
                current_uncovered = neighbor_uncovered

                # Track global best
                if current_cost < best_cost:
                    best_locations = list(current_locations)
                    best_cost = current_cost
                    best_coverage = current_coverage
                    best_uncovered = current_uncovered

            total_iters += 1

            # Early exit if target reached
            if best_coverage >= target_coverage:
                break

        history.append((T, best_cost, best_coverage, len(best_locations)))

        if verbose:
            print(f"{T:10.4f} {best_cost:10.4f} {best_coverage:10.2f}% {len(best_locations):5d}")

        # Early exit if target reached
        if best_coverage >= target_coverage:
            if verbose:
                print(f"\n*** Target coverage {target_coverage}% reached! ***")
            break

        # Cool down temperature
        if T > T_min:
            T *= alpha
        else:
            # Below T_min but target not reached - use low temperature for exploration
            T = T_min

    elapsed = time.time() - t_start

    if verbose:
        status = "MET" if best_coverage >= target_coverage else "NOT MET"
        if total_iters >= max_total_iters:
            status += " (SAFETY LIMIT REACHED)"
        print(f"\n{'='*60}")
        print(f"RESULT  —  Target {target_coverage}% [{status}]")
        print(f"  Coverage:        {best_coverage:.2f}%")
        print(f"  Base stations:   {len(best_locations)}")
        print(f"  Total iterations: {total_iters}")
        print(f"  Elapsed time:    {elapsed:.1f}s")
        print(f"  Cost:            {best_cost:.4f}")
        print(f"  Random seed:     {seed}")
        print(f"{'='*60}")
        print(f"\nBase Station Locations:")
        for i, (x, y) in enumerate(best_locations):
            print(f"  BS {i+1}: ({x:.1f}, {y:.1f}) m")

    return {
        'locations': best_locations,
        'num_bs': len(best_locations),
        'coverage': best_coverage,
        'cost': best_cost,
        'target_coverage': target_coverage,
        'history': history,
        'seed': seed,
    }


def run_sa_optimization(target_coverage: float = 95.0):
    """Run simulated annealing for a given coverage target and display the result."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import random
    import time

    # Generate random seed based on current time
    # random_seed = random.randint(1, 1000000)
    random_seed = 50

    print(f"\n{'='*60}")
    print(f"Random Seed: {random_seed}")
    print(f"{'='*60}\n")

    calc = CoverageCalculator("usc_map_buildings_filled.png")

    result = simulated_annealing(
        calc,
        target_coverage=target_coverage,
        initial_num_bs=8,
        seed=random_seed,
    )

    # --- Visualization ---
    detail = calc.calculate_coverage_detailed(result['locations'])
    image_data = detail['image_data']
    bs_colors = detail['bs_colors']
    small_h, small_w = image_data.shape[:2]

    # Upscale for display
    display_img = np.zeros((calc.img_height, calc.img_width, 3), dtype=np.float32)
    for y in range(calc.img_height):
        sy = min(y // COMPUTE_SCALE, small_h - 1)
        for x in range(calc.img_width):
            sx = min(x // COMPUTE_SCALE, small_w - 1)
            display_img[y, x] = image_data[sy, sx, :3] / 255.0

    colors_float = [(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in bs_colors]
    for i, (bs_x, bs_y) in enumerate(result['locations']):
        px = int(bs_x / MAP_WIDTH_M * calc.img_width)
        py = int(bs_y / MAP_HEIGHT_M * calc.img_height)
        color = colors_float[i] if i < len(colors_float) else (1.0, 0.0, 0.0)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= py+dy < calc.img_height and 0 <= px+dx < calc.img_width:
                    display_img[py+dy, px+dx] = color
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if abs(dy) == 4 or abs(dx) == 4:
                    if 0 <= py+dy < calc.img_height and 0 <= px+dx < calc.img_width:
                        display_img[py+dy, px+dx] = [0, 0, 0]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(display_img)

    cov = result['coverage']
    n = result['num_bs']
    random_seed = result['seed']
    status = "MET" if cov >= target_coverage else "NOT MET"
    ax.set_title(f"Target {target_coverage:.0f}% — {cov:.1f}% with {n} BSs [{status}] (Seed: {random_seed})")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    legend_patches = [mpatches.Patch(color=colors_float[i], label=f'BS {i+1}')
                      for i in range(n)]
    # Place legend outside plot area on the right
    ax.legend(handles=legend_patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)

    plt.suptitle("Simulated Annealing — Optimized BS Placement", fontsize=14)
    plt.tight_layout()
    output_filename = f'sa_optimization_seed{random_seed}.png'
    plt.savefig(output_filename, dpi=150)
    print(f"\nSaved SA visualization to: {output_filename}")
    plt.show()


# =============================================================================
# STANDALONE SIMULATION (for backwards compatibility)
# =============================================================================

def run_coverage_simulation():
    """Run standalone coverage simulation with visualization.

    This provides backwards compatibility with the original script behavior.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("=" * 60)
    print("USC Campus Coverage Simulation (Webapp-Compatible)")
    print("=" * 60)

    # Default parameters
    num_base_stations = 8

    print(f"\nSimulation Parameters:")
    print(f"  Transmit Power:     {TX_POWER_DBM} dBm")
    print(f"  Noise Floor:        {NOISE_DBM} dBm")
    print(f"  SNR Threshold:      {SNR_THRESHOLD_DB} dB")
    print(f"  Shadowing Std Dev:  {SHADOW_STD_DB} dB")
    print(f"  Carrier Frequency:  {FREQ_HZ/1e9} GHz")
    print(f"  Number of BSs:      {num_base_stations}")
    print(f"  Compute Scale:      1/{COMPUTE_SCALE}")
    print(f"  Num Rays (LOS):     {NUM_RAYS}")

    # Create calculator
    calc = CoverageCalculator("usc_map_buildings_filled.png")

    # Calculate max range
    max_range = calc.get_max_range()
    print(f"\nMax Coverage Range:   {max_range:.1f} meters")

    # Generate random base station locations
    rng = SeededRNG(SEED)
    base_stations = []
    for _ in range(num_base_stations):
        x_m, y_m = calc.get_random_outdoor_location(rng)
        base_stations.append((x_m, y_m))

    print(f"\nBase Station Locations:")
    for i, (x, y) in enumerate(base_stations):
        print(f"  BS {i+1}: ({x:.1f}, {y:.1f}) meters")

    # Calculate coverage
    print(f"\nCalculating coverage...")
    result = calc.calculate_coverage_detailed(base_stations)

    coverage_percent = result['coverage_percent']
    image_data = result['image_data']
    bs_colors = result['bs_colors']

    # Upscale image for display
    small_h, small_w = image_data.shape[:2]
    display_img = np.zeros((calc.img_height, calc.img_width, 3), dtype=np.float32)

    # Simple nearest-neighbor upscale
    for y in range(calc.img_height):
        sy = min(y // COMPUTE_SCALE, small_h - 1)
        for x in range(calc.img_width):
            sx = min(x // COMPUTE_SCALE, small_w - 1)
            display_img[y, x] = image_data[sy, sx, :3] / 255.0

    # Draw base station markers
    colors_float = [(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in bs_colors]
    for i, (bs_x, bs_y) in enumerate(base_stations):
        px = int(bs_x / MAP_WIDTH_M * calc.img_width)
        py = int(bs_y / MAP_HEIGHT_M * calc.img_height)
        color = colors_float[i] if i < len(colors_float) else (1.0, 0.0, 0.0)

        # Draw colored square
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= py+dy < calc.img_height and 0 <= px+dx < calc.img_width:
                    display_img[py+dy, px+dx] = color

        # Draw black border
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if abs(dy) == 4 or abs(dx) == 4:
                    if 0 <= py+dy < calc.img_height and 0 <= px+dx < calc.img_width:
                        display_img[py+dy, px+dx] = [0, 0, 0]

    # Display
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(display_img)

    # Create legend
    legend_patches = [mpatches.Patch(color=colors_float[i], label=f'BS {i+1}')
                      for i in range(num_base_stations)]
    ax.legend(handles=legend_patches, loc='upper right')

    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'USC Campus Coverage (Webapp-Compatible)\n'
                 f'(Tx={TX_POWER_DBM} dBm, Shadowing={SHADOW_STD_DB} dB)')

    plt.tight_layout()
    plt.savefig('output.png', dpi=150)
    print(f"\nSaved visualization to: output.png")
    plt.show()

    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print(f"=" * 60)
    print(f"Small grid size:       {small_w} x {small_h}")
    print(f"Outdoor pixels:        {calc.small_outdoor_count}")
    print(f"Coverage percentage:   {coverage_percent:.1f}%")
    print(f"=" * 60)


if __name__ == "__main__":
    run_sa_optimization()
