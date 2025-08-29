#!/usr/bin/env python3
"""
Gaze Visualization Utilities for Kivy-based G3 Controller

This module provides real-time gaze visualization capabilities including
heatmaps, foveated views, scanpaths, and opacity overlays that work with
Kivy's graphics system.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
import time
import math
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from scipy.ndimage import gaussian_filter
import logging

logger = logging.getLogger(__name__)


class GazeVisualizer:
    """
    Real-time gaze visualization system optimized for Kivy applications.
    
    Supports multiple visualization modes:
    - Fixation heatmaps with temporal decay
    - Foveated visualization with blur effects
    - Scanpath visualization with connections
    - Opacity-based attention overlays
    """
    
    def __init__(self, width: int, height: int, config: Optional[Dict] = None):
        """
        Initialize the gaze visualizer.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            config: Configuration dictionary for visualization parameters
        """
        self.width = width
        self.height = height
        
        # Default configuration
        self.config = {
            'heatmap': {
                'sigma': 25.0,
                'decay_rate': 0.98,
                'max_intensity': 1.0,
                'colormap': 'hot'
            },
            'foveated': {
                'fovea_radius': 80,
                'blur_strength': 15,
                'transition_width': 40
            },
            'scanpath': {
                'max_points': 100,
                'line_thickness': 3,
                'point_radius': 8,
                'fade_duration': 5.0
            },
            'opacity': {
                'sigma': 30.0,
                'max_opacity': 0.8,
                'gamma': 1.2
            }
        }
        
        # Update with user config
        if config:
            self._update_config(config)
        
        # Initialize data structures
        self.gaze_history = deque(maxlen=self.config['scanpath']['max_points'])
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.last_update_time = time.time()
        
        # Pre-generate color maps for performance
        self._initialize_colormaps()
        
        logger.info(f"GazeVisualizer initialized: {width}x{height}")
    
    def _update_config(self, config: Dict):
        """Recursively update configuration."""
        for key, value in config.items():
            if key in self.config and isinstance(value, dict):
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def _initialize_colormaps(self):
        """Pre-generate color maps for efficient rendering."""
        # Generate hot colormap (256 values)
        self.hot_colormap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            # Hot colormap: black -> red -> yellow -> white
            if i < 85:
                # Black to red
                self.hot_colormap[i] = [i * 3, 0, 0]
            elif i < 170:
                # Red to yellow
                self.hot_colormap[i] = [255, (i - 85) * 3, 0]
            else:
                # Yellow to white
                self.hot_colormap[i] = [255, 255, (i - 170) * 3]
    
    def update_gaze_data(self, gaze_data: Dict[str, Any], timestamp: float):
        """
        Update gaze data and internal state.
        
        Args:
            gaze_data: Dictionary containing gaze information
            timestamp: Timestamp of the gaze data
        """
        if 'gaze2d' not in gaze_data or not gaze_data['gaze2d']:
            return
        
        gaze_point = gaze_data['gaze2d']
        if len(gaze_point) != 2:
            return
        
        # Convert normalized coordinates to pixel coordinates
        x = int(gaze_point[0] * self.width)
        y = int(gaze_point[1] * self.height)
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        # Add to gaze history
        self.gaze_history.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        
        # Update heatmap with temporal decay
        self._update_heatmap(x, y, timestamp)
        
        logger.debug(f"Updated gaze data: ({x}, {y}) at {timestamp}")
    
    def _update_heatmap(self, x: int, y: int, timestamp: float):
        """Update the heatmap with temporal decay."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Apply temporal decay
        decay_factor = math.exp(-dt / self.config['heatmap']['decay_rate'])
        self.heatmap *= decay_factor
        
        # Add new fixation point
        if 0 <= x < self.width and 0 <= y < self.height:
            # Create a Gaussian blob around the fixation point
            sigma = self.config['heatmap']['sigma']
            size = int(sigma * 3)  # 3-sigma coverage
            
            # Create grid for Gaussian
            xx, yy = np.meshgrid(
                np.arange(max(0, x - size), min(self.width, x + size + 1)),
                np.arange(max(0, y - size), min(self.height, y + size + 1))
            )
            
            # Calculate Gaussian weights
            gaussian = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            
            # Add to heatmap
            y1, y2 = max(0, y - size), min(self.height, y + size + 1)
            x1, x2 = max(0, x - size), min(self.width, x + size + 1)
            
            if x2 > x1 and y2 > y1:
                self.heatmap[y1:y2, x1:x2] += gaussian
        
        # Normalize to prevent overflow
        if self.heatmap.max() > self.config['heatmap']['max_intensity']:
            self.heatmap = (self.heatmap / self.heatmap.max()) * self.config['heatmap']['max_intensity']
        
        self.last_update_time = current_time
    
    def create_fixation_heatmap(self) -> np.ndarray:
        """
        Create a fixation heatmap visualization.
        
        Returns:
            RGB image array with heatmap overlay
        """
        # Smooth the heatmap
        smoothed = gaussian_filter(self.heatmap, sigma=self.config['heatmap']['sigma'] / 3)
        
        # Normalize to 0-255 range
        if smoothed.max() > 0:
            normalized = (smoothed / smoothed.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(smoothed, dtype=np.uint8)
        
        # Apply colormap
        colored = self.hot_colormap[normalized]
        
        return colored
    
    def create_foveated_visualization(self, base_frame: np.ndarray) -> np.ndarray:
        """
        Create a foveated visualization with blur outside fixation areas.
        
        Args:
            base_frame: Base frame to apply foveation to
            
        Returns:
            Foveated frame with clear center and blurred periphery
        """
        if len(self.gaze_history) == 0:
            return base_frame.copy()
        
        # Get the most recent gaze point
        latest_gaze = self.gaze_history[-1]
        center_x, center_y = latest_gaze['x'], latest_gaze['y']
        
        # Create blur mask
        blur_mask = np.ones((self.height, self.width), dtype=np.float32)
        
        # Create distance map from fovea center
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        
        # Define fovea parameters
        fovea_radius = self.config['foveated']['fovea_radius']
        transition_width = self.config['foveated']['transition_width']
        
        # Create smooth transition from clear to blurred
        blur_mask = np.clip((distances - fovea_radius) / transition_width, 0, 1)
        
        # Create blurred version of the frame
        blur_strength = self.config['foveated']['blur_strength']
        blurred_frame = gaussian_filter(base_frame, sigma=blur_strength)
        
        # Blend based on blur mask
        result = base_frame.copy().astype(np.float32)
        for c in range(3):  # RGB channels
            result[:, :, c] = (
                base_frame[:, :, c] * (1 - blur_mask) +
                blurred_frame[:, :, c] * blur_mask
            )
        
        return result.astype(np.uint8)
    
    def create_scanpath_visualization(self) -> np.ndarray:
        """
        Create a scanpath visualization showing gaze trajectory.
        
        Returns:
            RGB image with scanpath overlay
        """
        # Create transparent overlay
        overlay = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        
        if len(self.gaze_history) < 2:
            return overlay[:, :, :3]  # Return RGB only
        
        current_time = time.time()
        fade_duration = self.config['scanpath']['fade_duration']
        line_thickness = self.config['scanpath']['line_thickness']
        point_radius = self.config['scanpath']['point_radius']
        
        # Draw connections between consecutive points
        points = list(self.gaze_history)
        for i in range(1, len(points)):
            p1, p2 = points[i-1], points[i]
            
            # Calculate age-based alpha
            age = current_time - p2['timestamp']
            alpha = max(0, 1 - age / fade_duration) * 255
            
            if alpha > 10:  # Only draw if reasonably visible
                # Draw line (simplified - in real implementation you'd use a proper line drawing algorithm)
                color = (0, 255, 0, int(alpha))  # Green with fade
                
                # For now, just draw points (lines would require more complex implementation)
                y, x = p2['y'], p2['x']
                if 0 <= x < self.width and 0 <= y < self.height:
                    # Draw point with radius
                    y_min = max(0, y - point_radius)
                    y_max = min(self.height, y + point_radius + 1)
                    x_min = max(0, x - point_radius)
                    x_max = min(self.width, x + point_radius + 1)
                    
                    overlay[y_min:y_max, x_min:x_max] = color
        
        return overlay[:, :, :3]  # Return RGB only
    
    def create_opacity_visualization(self, base_frame: np.ndarray) -> np.ndarray:
        """
        Create an opacity visualization where gazed areas are more/less opaque.
        
        Args:
            base_frame: Base frame to apply opacity effects to
            
        Returns:
            Frame with opacity-based attention overlay
        """
        if len(self.gaze_history) == 0:
            return base_frame.copy()
        
        # Create density map from recent gaze points
        density = np.zeros((self.height, self.width), dtype=np.float32)
        current_time = time.time()
        
        for gaze_point in self.gaze_history:
            age = current_time - gaze_point['timestamp']
            if age < 10.0:  # Only consider recent points
                weight = math.exp(-age / 2.0)  # Exponential decay
                x, y = gaze_point['x'], gaze_point['y']
                if 0 <= x < self.width and 0 <= y < self.height:
                    density[y, x] += weight
        
        # Smooth density
        sigma = self.config['opacity']['sigma']
        density = gaussian_filter(density, sigma=sigma)
        
        # Normalize and apply gamma correction
        if density.max() > 0:
            density = density / density.max()
        
        gamma = self.config['opacity']['gamma']
        density = np.power(density, gamma)
        
        # Convert to opacity (more attention = less opacity for "see-through" effect)
        max_opacity = self.config['opacity']['max_opacity']
        alpha = max_opacity * (1.0 - density)
        
        # Create semi-transparent overlay
        overlay = np.ones_like(base_frame, dtype=np.float32) * 128  # Gray overlay
        
        # Apply alpha blending
        result = base_frame.copy().astype(np.float32)
        for c in range(3):  # RGB channels
            result[:, :, c] = (
                base_frame[:, :, c] * (1 - alpha) +
                overlay[:, :, c] * alpha
            )
        
        return result.astype(np.uint8)
    
    def create_visualization_matrix(self, base_frame: np.ndarray) -> np.ndarray:
        """
        Create a 2x2 matrix showing all visualization types.
        
        Args:
            base_frame: Base frame for visualizations
            
        Returns:
            Combined visualization matrix
        """
        # Create individual visualizations
        heatmap = self.create_fixation_heatmap()
        foveated = self.create_foveated_visualization(base_frame)
        scanpath = self.create_scanpath_visualization()
        opacity = self.create_opacity_visualization(base_frame)
        
        # Resize each to half size
        half_h, half_w = self.height // 2, self.width // 2
        
        heatmap_small = np.array([
            [heatmap[i*2, j*2] for j in range(half_w)]
            for i in range(half_h)
        ], dtype=np.uint8)
        
        foveated_small = np.array([
            [foveated[i*2, j*2] for j in range(half_w)]
            for i in range(half_h)
        ], dtype=np.uint8)
        
        scanpath_small = np.array([
            [scanpath[i*2, j*2] for j in range(half_w)]
            for i in range(half_h)
        ], dtype=np.uint8)
        
        opacity_small = np.array([
            [opacity[i*2, j*2] for j in range(half_w)]
            for i in range(half_h)
        ], dtype=np.uint8)
        
        # Combine into 2x2 matrix
        top_row = np.hstack([heatmap_small, foveated_small])
        bottom_row = np.hstack([scanpath_small, opacity_small])
        matrix = np.vstack([top_row, bottom_row])
        
        return matrix
    
    def create_kivy_texture(self, image_array: np.ndarray) -> Texture:
        """
        Convert numpy array to Kivy texture for display.
        
        Args:
            image_array: RGB image array
            
        Returns:
            Kivy Texture object
        """
        # Ensure correct format
        if len(image_array.shape) == 3:
            h, w, c = image_array.shape
            if c == 3:
                # Convert RGB to RGBA
                rgba_array = np.zeros((h, w, 4), dtype=np.uint8)
                rgba_array[:, :, :3] = image_array
                rgba_array[:, :, 3] = 255  # Full alpha
                image_array = rgba_array
        
        # Flip vertically for Kivy (OpenGL coordinates)
        image_array = np.flipud(image_array)
        
        # Create texture
        texture = Texture.create(size=(image_array.shape[1], image_array.shape[0]))
        texture.blit_buffer(image_array.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        
        return texture
    
    def reset(self):
        """Reset all visualization data."""
        self.gaze_history.clear()
        self.heatmap.fill(0)
        self.last_update_time = time.time()
        logger.info("GazeVisualizer reset")


# Utility functions for integration with Kivy graphics
def apply_heatmap_to_texture(texture: Texture, heatmap: np.ndarray, blend_factor: float = 0.5) -> Texture:
    """
    Apply heatmap overlay to an existing Kivy texture.
    
    Args:
        texture: Original texture
        heatmap: Heatmap array
        blend_factor: Blending factor (0.0 = original, 1.0 = full heatmap)
        
    Returns:
        New texture with heatmap overlay
    """
    # This would be implemented to blend textures directly in GPU memory
    # For now, return original texture
    return texture


def create_colormap_texture(colormap_name: str = 'hot', size: int = 256) -> Texture:
    """
    Create a texture containing a colormap for GPU-based visualization.
    
    Args:
        colormap_name: Name of the colormap
        size: Size of the colormap texture
        
    Returns:
        Colormap texture
    """
    # Create colormap array
    if colormap_name == 'hot':
        colormap = np.zeros((1, size, 3), dtype=np.uint8)
        for i in range(size):
            if i < size // 3:
                colormap[0, i] = [i * 3 * 255 // size, 0, 0]
            elif i < 2 * size // 3:
                colormap[0, i] = [255, (i - size // 3) * 3 * 255 // size, 0]
            else:
                colormap[0, i] = [255, 255, (i - 2 * size // 3) * 3 * 255 // size]
    
    # Create texture
    texture = Texture.create(size=(size, 1))
    texture.blit_buffer(colormap.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
    
    return texture
