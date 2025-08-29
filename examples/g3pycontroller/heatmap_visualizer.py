import time
import logging
from collections import deque
from typing import Optional

import cv2
import numpy as np
from kivy.graphics.texture import Texture


class HeatmapVisualizer:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        
        # Heatmap configuration
        self.sigma = 35.0  # Gaussian blur sigma (reduced from 40 for better performance)
        self.max_intensity = 1.0
        self.blend_alpha = 0.5
        
        # Sliding window configuration
        self.max_gaze_points = 30
        self.gaze_history = deque(maxlen=self.max_gaze_points)
        
        # Initialize heatmap data
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Initialize colormap
        self._initialize_colormap()
        
        # Performance tracking
        self.frame_skip_counter = 0
        self.update_every_n_frames = 2
        self.decay_factor = 0.98
        
        # Control state
        self.is_enabled = True
        
        # Performance optimization: Pre-compute Gaussian kernel
        self._gaussian_kernel = None
        self._kernel_size = 0
        self._precompute_gaussian_kernel()
        
        # Performance monitoring
        self._update_times = deque(maxlen=10)
        self._last_update_time = 0
        
        logging.info(f"HeatmapVisualizer initialized: {width}x{height} with optimized Gaussian blur")
    
    def _initialize_colormap(self):
        """Initialize red-orange colormap for heatmap visualization."""
        self.hot_colormap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            if i < 64:
                self.hot_colormap[i] = [i * 4, 0, 0]
            elif i < 128:
                self.hot_colormap[i] = [255, (i - 64) * 4, 0]
            elif i < 192:
                self.hot_colormap[i] = [255, 255, 0]
            else:
                self.hot_colormap[i] = [255, 255, 0]
    
    def _precompute_gaussian_kernel(self):
        """Pre-compute Gaussian kernel for better performance."""
        try:
            # Calculate optimal kernel size
            self._kernel_size = int(self.sigma * 6)  # Reduced from 6 to 3 for performance
            if self._kernel_size % 2 == 0:
                self._kernel_size += 1
            
            # Pre-compute the Gaussian kernel
            center = self._kernel_size // 2
            y_grid, x_grid = np.ogrid[-center:center+1, -center:center+1]
            dist_sq = x_grid**2 + y_grid**2
            self._gaussian_kernel = np.exp(-dist_sq / (2 * self.sigma**2))
            
            # Normalize kernel
            self._gaussian_kernel /= self._gaussian_kernel.sum()
            
            logging.debug(f"Pre-computed Gaussian kernel: {self._kernel_size}x{self._kernel_size}, sigma={self.sigma}")
            
        except Exception as e:
            logging.error(f"Failed to precompute Gaussian kernel: {e}")
            self._gaussian_kernel = None
    
    def enable_heatmap(self):
        self.is_enabled = True
        logging.info("Heatmap visualization enabled")
    
    def disable_heatmap(self):
        self.is_enabled = False
        logging.info("Heatmap visualization disabled at timestamp: %s", time.time())
    
    def set_sliding_window_size(self, size: int):
        self.max_gaze_points = max(10, min(size, 50))
        current_points = list(self.gaze_history)
        self.gaze_history = deque(current_points[-self.max_gaze_points:], maxlen=self.max_gaze_points)
        logging.info(f"Sliding window size updated to {self.max_gaze_points} points")
    
    def update_gaze_data(self, gaze_point, timestamp):
        """Update heatmap with new gaze data using optimized Gaussian kernel."""
        if not self.is_enabled or gaze_point is None or len(gaze_point) != 2:
            return
        
        start_time = time.time()
        
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.update_every_n_frames:
            return
        self.frame_skip_counter = 0
        
        # Convert normalized coordinates to heatmap pixel coordinates
        x = int(gaze_point[0] * self.width)
        y = int((1 - gaze_point[1]) * self.height)
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        # Apply decay (more efficient in-place operation)
        np.multiply(self.heatmap, self.decay_factor, out=self.heatmap)
        
        # Add new gaze point using optimized method
        self._add_optimized_gaze_point(x, y)
        
        # Keep track of gaze history
        self.gaze_history.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        
        # Clamp intensities (in-place for better performance)
        np.clip(self.heatmap, 0, self.max_intensity, out=self.heatmap)
        
        # Track performance
        update_time = time.time() - start_time
        self._update_times.append(update_time)
        self._last_update_time = update_time
    
    def _add_optimized_gaze_point(self, x, y):
        """Add a gaze point using pre-computed Gaussian kernel (much faster)."""
        try:
            # Use pre-computed kernel if available, otherwise fall back to cv2
            if self._gaussian_kernel is not None:
                self._add_precomputed_gaze_point(x, y)
            else:
                # Fallback to cv2 method
                self._add_cv2_gaze_point(x, y)
                
        except Exception as e:
            logging.error(f"Failed to apply optimized Gaussian: {e}")
            # Final fallback to manual method
            self._add_fallback_gaze_point(x, y)
    
    def _add_precomputed_gaze_point(self, x, y):
        """Add gaze point using pre-computed kernel (fastest method)."""
        kernel_half = self._kernel_size // 2
        
        # Calculate bounds for kernel application
        y1 = max(0, y - kernel_half)
        y2 = min(self.height, y + kernel_half + 1)
        x1 = max(0, x - kernel_half)
        x2 = min(self.width, x + kernel_half + 1)
        
        # Calculate corresponding kernel bounds
        ky1 = max(0, kernel_half - y)
        ky2 = ky1 + (y2 - y1)
        kx1 = max(0, kernel_half - x)
        kx2 = kx1 + (x2 - x1)
        
        # Apply pre-computed kernel directly to heatmap
        self.heatmap[y1:y2, x1:x2] += self._gaussian_kernel[ky1:ky2, kx1:kx2]
    
    def _add_cv2_gaze_point(self, x, y):
        """Add a gaze point using cv2.GaussianBlur (slower but reliable)."""
        # Create a single-point intensity map
        point_map = np.zeros((self.height, self.width), dtype=np.float32)
        point_map[y, x] = 1.0  # Place a single intensity point
        
        # Apply Gaussian blur to the point
        ksize = self._kernel_size if self._kernel_size > 0 else int(self.sigma * 3)
        ksize = ksize if ksize % 2 == 1 else ksize + 1
        blurred = cv2.GaussianBlur(point_map, (ksize, ksize), self.sigma)
        
        # Add blurred point to heatmap
        self.heatmap += blurred
    
    def _add_fallback_gaze_point(self, x, y):
        """Fallback method using manual Gaussian kernel application."""
        kernel_size = int(self.sigma * 1.5)
        kernel = np.zeros((2 * kernel_size + 1, 2 * kernel_size + 1), dtype=np.float32)
        center = kernel_size
        y_grid, x_grid = np.ogrid[-center:center+1, -center:center+1]
        dist_sq = x_grid**2 + y_grid**2
        kernel = np.exp(-dist_sq / (2 * self.sigma**2))
        
        y1 = max(0, y - kernel_size)
        y2 = min(self.height, y + kernel_size + 1)
        x1 = max(0, x - kernel_size)
        x2 = min(self.width, x + kernel_size + 1)
        
        ky1 = max(0, kernel_size - y)
        ky2 = ky1 + (y2 - y1)
        kx1 = max(0, kernel_size - x)
        kx2 = kx1 + (x2 - x1)
        
        try:
            self.heatmap[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
        except (ValueError, IndexError):
            pass
    
    def create_heatmap_overlay(self, video_frame):
        """Create heatmap overlay blended with video frame."""
        if not self.is_enabled:
            return video_frame
        
        heatmap_max = self.heatmap.max()
        if heatmap_max <= 1e-6:
            return video_frame
        
        frame_height, frame_width = video_frame.shape[:2]
        normalized_heatmap = self.heatmap.copy() if heatmap_max != 1.0 else self.heatmap
        if heatmap_max != 1.0:
            normalized_heatmap /= heatmap_max
        
        if normalized_heatmap.shape != (frame_height, frame_width):
            try:
                # import cv2
                normalized_heatmap = cv2.resize(normalized_heatmap, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                y_ratio = self.height / frame_height
                x_ratio = self.width / frame_width
                y_indices = (np.arange(frame_height) * y_ratio).astype(np.int32)
                x_indices = (np.arange(frame_width) * x_ratio).astype(np.int32)
                y_indices = np.clip(y_indices, 0, self.height - 1)
                x_indices = np.clip(x_indices, 0, self.width - 1)
                normalized_heatmap = normalized_heatmap[np.ix_(y_indices, x_indices)]
        
        mask = normalized_heatmap > 0.01
        if not mask.any():
            return video_frame
        
        heatmap_indices = (normalized_heatmap * 255).astype(np.uint8)
        colored_heatmap = self.hot_colormap[heatmap_indices]
        
        alpha = (normalized_heatmap * self.blend_alpha)[..., np.newaxis]
        blend_mask = alpha[..., 0] > 0.01
        if not blend_mask.any():
            return video_frame
        
        result = video_frame.copy()
        inv_alpha = 1 - alpha
        result[blend_mask] = (
            video_frame[blend_mask].astype(np.float32) * inv_alpha[blend_mask] +
            colored_heatmap[blend_mask].astype(np.float32) * alpha[blend_mask]
        ).astype(np.uint8)
        
        return result
    
    def create_heatmap_texture(self):
        """Create standalone heatmap texture for debugging/saving."""
        if self.heatmap.max() == 0:
            return None
        
        normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        colored = self.hot_colormap[normalized]
        
        rgba_heatmap = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba_heatmap[:, :, :3] = colored
        rgba_heatmap[:, :, 3] = (normalized * self.blend_alpha).astype(np.uint8)
        
        rgba_heatmap = np.flipud(rgba_heatmap)
        
        texture = Texture.create(size=(self.width, self.height))
        texture.blit_buffer(rgba_heatmap.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        
        return texture
    
    def reset(self):
        """Reset heatmap data."""
        self.heatmap.fill(0)
        self.gaze_history.clear()
        logging.info("HeatmapVisualizer reset")
    
    def set_performance_mode(self, mode="balanced"):
        """Set performance mode for heatmap visualization."""
        if mode == "fast":
            self.update_every_n_frames = 3
            self.max_gaze_points = 15
            self.sigma = 25.0
            self.decay_factor = 0.95
        elif mode == "quality":
            self.update_every_n_frames = 1
            self.max_gaze_points = 50
            self.sigma = 50.0
            self.decay_factor = 0.99
        else:  # balanced
            self.update_every_n_frames = 15  # Fixed from 25 to 2 for much better performance
            self.max_gaze_points = 50
            self.sigma = 55.0
            self.decay_factor = 0.98
        
        # Recompute kernel with new sigma
        self._precompute_gaussian_kernel()
        
        logging.info(f"Performance mode set to: {mode}")
    
    def set_update_frequency(self, frames_to_skip):
        self.update_every_n_frames = max(1, min(frames_to_skip, 10))
        logging.info(f"Heatmap update frequency set to every {self.update_every_n_frames} frames")
    
    def set_blur_radius(self, radius):
        self.sigma = max(10.0, min(radius, 100.0))
        # Recompute kernel with new sigma
        self._precompute_gaussian_kernel()
        logging.info(f"Heatmap blur radius set to: {self.sigma}")
    
    def get_performance_stats(self):
        avg_update_time = sum(self._update_times) / len(self._update_times) if self._update_times else 0
        return {
            'gaze_points_in_history': len(self.gaze_history),
            'max_gaze_points': self.max_gaze_points,
            'heatmap_max_intensity': float(self.heatmap.max()),
            'is_enabled': self.is_enabled,
            'update_frequency': self.update_every_n_frames,
            'blur_radius': self.sigma,
            'decay_factor': self.decay_factor,
            'kernel_size': self._kernel_size,
            'has_precomputed_kernel': self._gaussian_kernel is not None,
            'last_update_time_ms': self._last_update_time * 1000,
            'avg_update_time_ms': avg_update_time * 1000,
            'update_times_count': len(self._update_times),
        }
