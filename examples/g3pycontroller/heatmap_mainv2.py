import asyncio
import json
import logging

import time
from collections import deque
from typing import List, Optional, Set, Tuple, cast

import aiohttp
import numpy as np
from eventkinds import AppEventKind, ControlEventKind
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.texture import Texture
from kivy.lang.builder import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.screenmanager import Screen, ScreenManager

from g3pylib import Glasses3, connect_to_glasses
from g3pylib.g3typing import SignalBody
from g3pylib.recordings import RecordingsEventKind
from g3pylib.recordings.recording import Recording
from g3pylib.zeroconf import EventKind, G3Service, G3ServiceDiscovery

GAZE_CIRCLE_RADIUS = 10
VIDEOPLAYER_PROGRESS_BAR_HEIGHT = dp(44)
VIDEO_Y_TO_X_RATIO = 9 / 16
LIVE_FRAME_RATE = 15

logging.basicConfig(level=logging.DEBUG)

# fmt: off
Builder.load_string("""
#:import NoTransition kivy.uix.screenmanager.NoTransition
#:import Factory kivy.factory.Factory
#:import ControlEventKind eventkinds.ControlEventKind
#:import AppEventKind eventkinds.AppEventKind

<DiscoveryScreen>:
    BoxLayout:
        BoxLayout:
            orientation: "vertical"
            Label:
                size_hint_y: None
                height: dp(50)
                text: "Found services:"
            SelectableList:
                id: services
        Button:
            size_hint: 1, None
            height: dp(50)
            pos_hint: {'center_x':0.5, 'center_y':0.5}
            text: "Connect"
            on_press: app.send_app_event(AppEventKind.ENTER_CONTROL_SESSION)

<UserMessagePopup>:
    size_hint: None, None
    size: 400, 200
    Label:
        id: message_label
        text: ""

<ControlScreen>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            size_hint: 1, None
            height: dp(50)
            Label:
                id: hostname
                text: "Hostname placeholder"
                halign: "left"
            Label:
                id: task_indicator
                text: ""
        BoxLayout:
            size_hint: 1, None
            height: dp(50)
            Button:
                text: "Recorder"
                on_press: root.switch_to_screen("recorder")
            Button:
                text: "Live"
                on_press: root.switch_to_screen("live")
            Button:
                background_color: (0.6, 0.6, 1, 1)
                text: "Disconnect"
                on_press:
                    app.send_app_event(AppEventKind.LEAVE_CONTROL_SESSION)
        ScreenManager:
            id: sm
            transition: NoTransition()

<RecordingScreen>:
    VideoPlayer:
        id: videoplayer

<RecorderScreen>:
    BoxLayout:
        BoxLayout:
            orientation: 'vertical'
            Label:
                id: recorder_status
                text: "Status:"
            Button:
                text: "Start"
                on_press: app.send_control_event(ControlEventKind.START_RECORDING)
            Button:
                text: "Stop"
                on_press: app.send_control_event(ControlEventKind.STOP_RECORDING)
            Button:
                text: "Delete"
                on_press: app.send_control_event(ControlEventKind.DELETE_RECORDING)
            Button:
                text: "Play"
                on_press: app.send_control_event(ControlEventKind.PLAY_RECORDING)
        SelectableList:
            id: recordings

<LiveScreen>:
    BoxLayout:
        Widget:
            id: display
            size_hint_x: 0.8
            size_hint_y: 1
        BoxLayout:
            orientation: "vertical"
            size_hint_x: 0.2
            Label:
                text: "Live Feed Controls"
                size_hint_y: None
                height: dp(30)
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(100)
                Button:
                    text: "Start Live"
                    on_press: app.send_control_event(ControlEventKind.START_LIVE)
                Button:
                    text: "Stop Live"
                    on_press: app.send_control_event(ControlEventKind.STOP_LIVE)
            Label:
                text: "Heatmap Overlay"
                size_hint_y: None
                height: dp(30)
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(150)
                Button:
                    id: heatmap_start_btn
                    text: "Start Heatmap"
                    on_press: app.send_control_event(ControlEventKind.START_HEATMAP)
                Button:
                    id: heatmap_stop_btn
                    text: "Stop Heatmap"
                    on_press: app.send_control_event(ControlEventKind.STOP_HEATMAP)
                Button:
                    text: "Reset Heatmap"
                    on_press: app.reset_heatmap()
            Widget:
                # Spacer

<SelectableList>:
    viewclass: 'SelectableLabel'
    SelectableRecycleBoxLayout:
        id: selectables
        default_size: None, dp(70)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'

<SelectableLabel>:
    canvas.before:
        Color:
            rgba: (.0, 0.9, .1, .3) if self.selected else (0, 0, 0, 1)
        Rectangle:
            pos: self.pos
            size: self.size
"""
)
# fmt: on


class SelectableRecycleBoxLayout(LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class SelectableLabel(RecycleDataViewBehavior, Label):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    def refresh_view_attrs(self, rv, index, data):
        """Catch and handle the view changes"""
        self.index = index
        return super().refresh_view_attrs(rv, index, data)

    def on_touch_down(self, touch):
        """Add selection on touch down"""
        if super().on_touch_down(touch):
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        """Respond to the selection of items in the view."""
        self.selected = is_selected


class SelectableList(RecycleView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []


class DiscoveryScreen(Screen):
    def add_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        self.ids.services.data.append(
            {"hostname": hostname, "text": f"{hostname}\n{ipv4}\n{ipv6}"}
        )
        logging.info(f"Services: Added {hostname}, {ipv4}, {ipv6}")

    def update_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        services = self.ids.services
        for service in services.data:
            if service["hostname"] == hostname:
                service["text"] = f"{hostname}\n{ipv4}\n{ipv6}"
                logging.info(f"Services: Updated {hostname}, {ipv4}, {ipv6}")

    def remove_service(
        self, hostname: str, ipv4: Optional[str], ipv6: Optional[str]
    ) -> None:
        services = self.ids.services
        services.data = [
            service for service in services.data if service["hostname"] != hostname
        ]
        logging.info(f"Services: Removed {hostname}, {ipv4}, {ipv6}")

    def clear(self):
        self.ids.services.data = []
        logging.info("Services: All cleared")


class ControlScreen(Screen):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.ids.sm.add_widget(RecorderScreen(name="recorder"))
        self.ids.sm.add_widget(RecordingScreen(name="recording"))
        self.ids.sm.add_widget(LiveScreen(name="live"))

    def clear(self) -> None:
        self.ids.sm.get_screen("recorder").ids.recordings.data = []
        self.ids.sm.get_screen("recorder").ids.recorder_status.text = "Status:"

    def switch_to_screen(self, screen: str) -> None:
        self.ids.sm.current = screen
        if self.ids.sm.current == "recording":
            self.ids.sm.get_screen("recording").ids.videoplayer.state = "stop"

    def set_task_running_status(self, is_running: bool) -> None:
        if is_running:
            self.ids.task_indicator.text = "Handling action..."
        else:
            self.ids.task_indicator.text = ""

    def set_hostname(self, hostname: str) -> None:
        self.ids.hostname.text = hostname


class RecordingScreen(Screen):
    pass


class RecorderScreen(Screen):
    def add_recording(
        self, visible_name: str, uuid: str, recording: Recording, atEnd: bool = False
    ) -> None:
        recordings = self.ids.recordings
        recording_data = {"text": visible_name, "uuid": uuid, "recording": recording}
        if atEnd:
            recordings.data.append(recording_data)
        else:
            recordings.data.insert(0, recording_data)

    def remove_recording(self, uuid: str) -> None:
        recordings = self.ids.recordings
        recordings.data = [rec for rec in recordings.data if rec["uuid"] != uuid]

    def set_recording_status(self, is_recording: bool) -> None:
        if is_recording:
            self.ids.recorder_status.text = "Status: Recording"
        else:
            self.ids.recorder_status.text = "Status: Not recording"


class LiveScreen(Screen):
    def clear(self, *args):
        self.ids.display.canvas.clear()


class UserMessagePopup(Popup):
    pass


class GazeCircle:
    def __init__(self, canvas, origin, size) -> None:
        self.canvas = canvas
        self.origin = origin
        self.size = size
        self.circle_obj = Line(circle=(0, 0, 0))
        self.canvas.add(self.circle_obj)

    def redraw(self, coord):
        self.canvas.remove(self.circle_obj)
        self.canvas.add(Color(1, 0, 0, 1))
        if coord is None:
            self.circle_obj = Line(circle=(0, 0, 0))
        else:
            circle_x = self.origin[0] + coord[0] * self.size[0]
            circle_y = self.origin[1] + (1 - coord[1]) * self.size[1]
            self.circle_obj = Line(circle=(circle_x, circle_y, GAZE_CIRCLE_RADIUS))
        self.canvas.add(self.circle_obj)
        self.canvas.remove(Color(1, 0, 0, 1))

    def reset(self):
        self.canvas.remove(self.circle_obj)
        self.circle_obj = Line(circle=(0, 0, 0))
        self.canvas.add(self.circle_obj)


class HeatmapVisualizer:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        
        # Heatmap configuration - Optimized for sliding window updates
        self.sigma = 55.0  # Adjusted for better visibility
        self.max_intensity = 1.0
        self.blend_alpha = 0.6  # Transparency for overlay
        
        # Sliding window configuration
        self.max_gaze_points = 50  # Keep last 50 gaze points
        self.rebuild_frequency = 50  # Rebuild heatmap every 50 new points
        self.gaze_point_counter = 0  # Counter for tracking when to rebuild
        # Performance optimization settings
        self.update_counter = 0
        self.update_frequency = 1  # Update display every frame for immediate response
        self.texture_cache_time = 0
        self.texture_cache_duration = 0.05  # Shorter cache for more responsive updates
        self.overlay_skip_frames = 1  # Process every frame for immediate display
        
        # Initialize heatmap data
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.last_update_time = time.time()
        self.gaze_history = deque(maxlen=self.max_gaze_points)  # Sliding window of gaze points
        
        # Caching for expensive operations
        self.cached_overlay = None
        self.cached_frame_shape = None
        self.cached_normalized_heatmap = None
        self.last_heatmap_hash = None
        
        # Pre-computed Gaussian templates for common sizes
        self._gaussian_cache = {}
        self._max_cache_size = 20
        
        # Initialize colormap for heatmap visualization
        self._initialize_colormap()
        
        # Initialize graphics objects
        self.heatmap_texture = None
        self.heatmap_rect = None
        self.cached_texture = None
        
        # Control state
        self.is_enabled = True  # Start with heatmap enabled by default
        
        # Performance monitoring
        self.frame_times = deque(maxlen=30)
        self.performance_mode = 'normal'  # 'normal', 'fast', 'ultra_fast'
        
        logging.info(f"HeatmapVisualizer initialized: {width}x{height} with performance optimizations")
    
    def _initialize_colormap(self):
        """Initialize hot colormap for heatmap visualization."""
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
    
    def enable_heatmap(self):
        """Enable heatmap visualization."""
        self.is_enabled = True
        logging.info("Heatmap visualization enabled")
    
    def disable_heatmap(self):
        """Disable heatmap visualization."""
        self.is_enabled = False
        logging.info("Heatmap visualization disabled")
    
    def set_sliding_window_size(self, size: int):
        """
        Update the sliding window size for gaze points.
        
        Args:
            size: Number of gaze points to keep in the sliding window
        """
        self.max_gaze_points = max(10, min(size, 200))  # Clamp between 10-200
        self.rebuild_frequency = self.max_gaze_points  # Rebuild when window is full
        
        # Update the deque with new maxlen
        current_points = list(self.gaze_history)
        self.gaze_history = deque(current_points[-self.max_gaze_points:], maxlen=self.max_gaze_points)
        
        # Reset counter to trigger rebuild with new window size
        self.gaze_point_counter = 0
        
        logging.info(f"Sliding window size updated to {self.max_gaze_points} points")
    
    def update_gaze_data(self, gaze_point, timestamp):
        """Update heatmap with new gaze data using sliding window approach."""
        if not self.is_enabled or gaze_point is None or len(gaze_point) != 2:
            return
        
        # Convert normalized coordinates to heatmap pixel coordinates
        # gaze_point is normalized (0-1), so scale to heatmap dimensions
        # Use same coordinate system as gaze circle for consistency
        x = int(gaze_point[0] * self.width)
        # For y-coordinate, flip to match video frame coordinate system (same as gaze circle)
        y = int((1 - gaze_point[1]) * self.height)
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        
        # Add to gaze history (automatically removes oldest if at max capacity)
        self.gaze_history.append({
            'x': x,
            'y': y,
            'timestamp': timestamp
        })
        
        # Increment counter and check if we need to rebuild heatmap
        self.gaze_point_counter += 1
        
        # Rebuild heatmap from scratch every N points to remove old data influence
        if self.gaze_point_counter >= self.rebuild_frequency:
            self._rebuild_heatmap_from_history()
            self.gaze_point_counter = 0  # Reset counter
            logging.debug(f"Heatmap rebuilt from {len(self.gaze_history)} recent gaze points")
        else:
            # Just add the new point to existing heatmap for performance
            self._add_gaze_point_to_heatmap(x, y)
    
    def _get_cached_gaussian(self, w, h, cx, cy):
        """Get or create cached Gaussian blob for given dimensions."""
        cache_key = (w, h, cx, cy)
        
        if cache_key in self._gaussian_cache:
            return self._gaussian_cache[cache_key]
        
        # Clear cache if it gets too large
        if len(self._gaussian_cache) >= self._max_cache_size:
            # Remove oldest entries (simple FIFO)
            for _ in range(5):
                if self._gaussian_cache:
                    self._gaussian_cache.pop(next(iter(self._gaussian_cache)))
        
        # Create new Gaussian
        yy = np.arange(h)[:, None]
        xx = np.arange(w)[None, :]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        gaussian = np.exp(-dist_sq / (self.sigma ** 2))
        
        self._gaussian_cache[cache_key] = gaussian
        return gaussian
    
    def _update_performance_mode(self):
        """Dynamically adjust performance mode based on frame processing times."""
        if len(self.frame_times) >= 10:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            
            # Adjust performance mode based on average processing time
            if avg_time > 0.05:  # > 50ms per frame (< 20 FPS)
                self.performance_mode = 'ultra_fast'
                self.overlay_skip_frames = 4
            elif avg_time > 0.033:  # > 33ms per frame (< 30 FPS)
                self.performance_mode = 'fast'
                self.overlay_skip_frames = 3
            else:
                self.performance_mode = 'normal'
                self.overlay_skip_frames = 2
    
    def _rebuild_heatmap_from_history(self):
        """Rebuild the entire heatmap from the current gaze history."""
        # Clear the current heatmap
        self.heatmap.fill(0)
        
        # Rebuild from all points in history
        for gaze_point in self.gaze_history:
            self._add_gaze_point_to_heatmap(gaze_point['x'], gaze_point['y'])
        
        # Normalize the rebuilt heatmap
        if self.heatmap.max() > self.max_intensity:
            self.heatmap = (self.heatmap / self.heatmap.max()) * self.max_intensity
        
        logging.debug(f"Heatmap rebuilt with max intensity: {self.heatmap.max():.4f}")
    
    def _add_gaze_point_to_heatmap(self, x, y):
        """Add a single gaze point to the heatmap using cached Gaussian blob."""
        if 0 <= x < self.width and 0 <= y < self.height:
            # Adaptive size based on performance mode
            size_multiplier = {'normal': 1.5, 'fast': 1.2, 'ultra_fast': 1.0}[self.performance_mode]
            size = int(self.sigma * size_multiplier)
            
            # Calculate bounds for the Gaussian
            y1, y2 = max(0, y - size), min(self.height, y + size + 1)
            x1, x2 = max(0, x - size), min(self.width, x + size + 1)
            
            if x2 > x1 and y2 > y1:
                w, h = x2 - x1, y2 - y1
                cy, cx = y - y1, x - x1
                
                # Use cached Gaussian for better performance
                gaussian = self._get_cached_gaussian(w, h, cx, cy)
                
                # Add intensity based on performance mode
                intensity = {'normal': 1.5, 'fast': 1.2, 'ultra_fast': 1.0}[self.performance_mode]
                self.heatmap[y1:y2, x1:x2] += gaussian * intensity
        
        # Periodic normalization to prevent overflow (less frequent than rebuilds)
        current_time = time.time()
        if current_time - self.last_update_time > 2.0:  # Every 2 seconds
            if self.heatmap.max() > self.max_intensity * 2.0:
                self.heatmap = (self.heatmap / self.heatmap.max()) * self.max_intensity
            self.last_update_time = current_time
    
    def create_heatmap_overlay(self, video_frame):
        """Create heatmap overlay blended with video frame - optimized version."""
        start_time = time.time()
        
        if not self.is_enabled:
            return video_frame
        
        # Check if heatmap has any content
        if self.heatmap.max() <= 0:
            return video_frame
        
        # Update counter for performance optimization
        self.update_counter += 1
        
        # Skip frames based on performance mode and counter
        if self.update_counter % self.overlay_skip_frames != 0:
            # Return cached overlay if available, otherwise original frame
            return self.cached_overlay if self.cached_overlay is not None else video_frame
        
        # Update performance mode periodically
        if self.update_counter % 30 == 0:  # Every 30 frames
            self._update_performance_mode()
        
        # Get video frame dimensions
        frame_height, frame_width = video_frame.shape[:2]
        frame_shape = (frame_height, frame_width)
        
        # Check if we can use cached normalized heatmap
        current_heatmap_hash = hash(self.heatmap.tobytes())
        use_cached_heatmap = (
            self.cached_normalized_heatmap is not None and
            self.last_heatmap_hash == current_heatmap_hash and
            self.cached_frame_shape == frame_shape
        )
        
        if use_cached_heatmap:
            normalized_heatmap = self.cached_normalized_heatmap
        else:
            # Normalize heatmap to 0-1 range
            normalized_heatmap = self.heatmap / self.heatmap.max()
            
        # Resize heatmap to match video frame dimensions if needed
        if normalized_heatmap.shape != frame_shape:
            # Use simple resizing to match frame dimensions - prefer fastest method
            try:
                import cv2
                # Use fastest interpolation for better performance
                interpolation = getattr(cv2, 'INTER_NEAREST', 0) if self.performance_mode == 'ultra_fast' else getattr(cv2, 'INTER_LINEAR', 1)
                normalized_heatmap = getattr(cv2, 'resize', lambda x, y, **kwargs: x)(normalized_heatmap, (frame_width, frame_height), interpolation=interpolation)
            except (ImportError, AttributeError):
                # Fallback to numpy-based nearest neighbor resizing for performance
                y_indices = np.linspace(0, self.height-1, frame_height).astype(int)
                x_indices = np.linspace(0, self.width-1, frame_width).astype(int)
                normalized_heatmap = normalized_heatmap[np.ix_(y_indices, x_indices)]
        
        # Cache the normalized heatmap for reuse
        self.cached_normalized_heatmap = normalized_heatmap
        self.last_heatmap_hash = current_heatmap_hash
        self.cached_frame_shape = frame_shape
        
        # Apply colormap with performance optimization
        if self.performance_mode == 'ultra_fast':
            # Skip colormap for ultra-fast mode, use grayscale
            colored_heatmap = np.stack([normalized_heatmap * 255] * 3, axis=-1).astype(np.uint8)
        else:
            heatmap_indices = (normalized_heatmap * 255).astype(np.uint8)
            colored_heatmap = self.hot_colormap[heatmap_indices]
        
        # Adaptive alpha blending based on performance mode
        alpha_multiplier = {'normal': 1.0, 'fast': 0.8, 'ultra_fast': 0.6}[self.performance_mode]
        alpha = normalized_heatmap * self.blend_alpha * alpha_multiplier
        
        # Ensure video frame is proper format
        if len(video_frame.shape) == 3 and video_frame.shape[2] == 3:
            # Optimized blending using vectorized operations
            if self.performance_mode == 'ultra_fast':
                # Ultra-fast mode: simple additive blending
                blended_frame = video_frame.astype(np.float32)
                blended_frame += colored_heatmap * alpha[..., np.newaxis] * 0.3
                blended_frame = np.clip(blended_frame, 0, 255).astype(np.uint8)
            else:
                # Normal blending for better quality
                inv_alpha = 1 - alpha
                blended_frame = (
                    video_frame.astype(np.float32) * inv_alpha[..., np.newaxis] +
                    colored_heatmap.astype(np.float32) * alpha[..., np.newaxis]
                ).astype(np.uint8)
            
            # Cache the result
            self.cached_overlay = blended_frame
            
            # Track frame processing time for performance monitoring
            processing_time = time.time() - start_time
            self.frame_times.append(processing_time)
            
            return blended_frame
        
        return video_frame
    
    def create_heatmap_texture(self):
        """Create standalone heatmap texture for debugging/saving."""
        if self.heatmap.max() == 0:
            return None
        
        # Normalize heatmap to 0-255 range
        normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        
        # Apply colormap
        colored = self.hot_colormap[normalized]
        
        # Add alpha channel for blending
        rgba_heatmap = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba_heatmap[:, :, :3] = colored
        rgba_heatmap[:, :, 3] = (normalized * self.blend_alpha).astype(np.uint8)
        
        # Flip for Kivy coordinate system
        rgba_heatmap = np.flipud(rgba_heatmap)
        
        # Create texture
        texture = Texture.create(size=(self.width, self.height))
        texture.blit_buffer(rgba_heatmap.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        
        return texture
    
    def _apply_simple_smoothing(self, heatmap):
        """Apply simple smoothing kernel instead of Gaussian."""
        # Simple 3x3 smoothing kernel
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
        
        # Apply convolution-like smoothing
        smoothed = np.copy(heatmap)
        h, w = heatmap.shape
        
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Apply kernel to 3x3 neighborhood
                neighborhood = heatmap[i-1:i+2, j-1:j+2]
                smoothed[i, j] = np.sum(neighborhood * kernel)
        
        return smoothed
    

    
    def clear_cache(self):
        """Clear all cached data for memory management."""
        self.cached_overlay = None
        self.cached_frame_shape = None
        self.cached_normalized_heatmap = None
        self.last_heatmap_hash = None
        self._gaussian_cache.clear()
        self.cached_texture = None
        
    def reset(self):
        """Reset heatmap data and clear caches."""
        self.heatmap.fill(0)
        self.gaze_history.clear()
        self.gaze_point_counter = 0  # Reset the gaze point counter
        self.last_update_time = time.time()
        
        # Reset performance counters and caches
        self.update_counter = 0
        self.texture_cache_time = 0
        self.frame_times.clear()
        self.performance_mode = 'normal'
        self.overlay_skip_frames = 2
        
        # Clear all caches
        self.clear_cache()
        
        logging.info("HeatmapVisualizer reset with sliding window configuration")
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        if len(self.frame_times) == 0:
            return {
                'avg_frame_time': 0,
                'fps_estimate': 0,
                'performance_mode': self.performance_mode,
                'cache_size': len(self._gaussian_cache),
                'frames_processed': self.update_counter,
                'skip_frames': self.overlay_skip_frames
            }
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        fps_estimate = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_frame_time': avg_time * 1000,  # Convert to milliseconds
            'fps_estimate': fps_estimate,
            'performance_mode': self.performance_mode,
            'cache_size': len(self._gaussian_cache),

            'frames_processed': self.update_counter,
            'skip_frames': self.overlay_skip_frames,
            'gaze_points_in_history': len(self.gaze_history),
            'gaze_point_counter': self.gaze_point_counter,
            'rebuild_frequency': self.rebuild_frequency
        }


class G3App(App, ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Window.bind(on_request_close=self.close)
        self.tasks: Set[asyncio.Task] = set()
        self.app_events: asyncio.Queue[AppEventKind] = asyncio.Queue()
        self.control_events: asyncio.Queue[ControlEventKind] = asyncio.Queue()
        self.live_stream_task: Optional[asyncio.Task] = None
        self.read_frames_task: Optional[asyncio.Task] = None
        self.add_widget(DiscoveryScreen(name="discovery"))
        self.add_widget(ControlScreen(name="control"))
        self.latest_frame_with_timestamp = None
        self.latest_gaze_with_timestamp = None
        self.live_gaze_circle = None
        self.live_heatmap_visualizer = None
        self.replay_gaze_circle = None
        self.last_texture = None
        self.draw_frame_event = None

    def build(self):
        return self

    def on_start(self):
        self.create_task(self.backend_app(), name="backend_app")
        self.send_app_event(AppEventKind.START_DISCOVERY)

    def close(self, *args) -> bool:
        self.send_app_event(AppEventKind.STOP)
        return True

    def switch_to_screen(self, screen: str):
        if screen == "discovery":
            self.transition.direction = "right"
        else:
            self.transition.direction = "left"
        self.current = screen

    def start_control(self) -> bool:
        selected = self.get_screen(
            "discovery"
        ).ids.services.ids.selectables.selected_nodes
        if len(selected) <= 0:
            popup = UserMessagePopup(title="No Glasses3 unit selected")
            popup.ids.message_label.text = (
                "Please select a Glasses3 unit and try again."
            )
            popup.open()
            return False
        else:
            hostname = self.get_screen("discovery").ids.services.data[selected[0]][
                "hostname"
            ]
            self.backend_control_task = self.create_task(
                self.backend_control(hostname), name="backend_control"
            )
            self.get_screen("control").set_hostname(hostname)
            self.switch_to_screen("control")
            return True

    async def stop_control(self) -> None:
        await self.cancel_task(self.backend_control_task)
        self.get_screen("control").clear()

    def start_discovery(self):
        self.discovery_task = self.create_task(
            self.backend_discovery(), name="backend_discovery"
        )
        self.switch_to_screen("discovery")

    async def stop_discovery(self):
        await self.cancel_task(self.discovery_task)
        self.get_screen("discovery").clear()

    def send_app_event(self, event: AppEventKind) -> None:
        self.app_events.put_nowait(event)

    async def backend_app(self) -> None:
        while True:
            app_event = await self.app_events.get()
            await self.handle_app_event(app_event)
            if app_event == AppEventKind.STOP:
                break

    async def handle_app_event(self, event: AppEventKind):
        logging.info(f"Handling app event: {event}")
        match event:
            case AppEventKind.START_DISCOVERY:
                self.start_discovery()
            case AppEventKind.ENTER_CONTROL_SESSION:
                if self.start_control():
                    await self.stop_discovery()
            case AppEventKind.LEAVE_CONTROL_SESSION:
                self.start_discovery()
                await self.stop_control()
            case AppEventKind.STOP:
                match self.current:
                    case "discovery":
                        await self.stop_discovery()
                    case "control":
                        await self.stop_control()
                self.stop()

    async def backend_discovery(self) -> None:
        async with G3ServiceDiscovery.listen() as service_listener:
            while True:
                await self.handle_service_event(await service_listener.events.get())

    async def handle_service_event(self, event: Tuple[EventKind, G3Service]) -> None:
        logging.info(f"Handling service event: {event[0]}")
        match event:
            case (EventKind.ADDED, service):
                self.get_screen("discovery").add_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )
            case (EventKind.UPDATED, service):
                self.get_screen("discovery").update_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )
            case (EventKind.REMOVED, service):
                self.get_screen("discovery").remove_service(
                    service.hostname, service.ipv4_address, service.ipv6_address
                )

    def send_control_event(self, event: ControlEventKind) -> None:
        self.control_events.put_nowait(event)

    async def backend_control(self, hostname: str) -> None:
        async with connect_to_glasses.with_hostname(hostname) as g3:
            async with g3.recordings.keep_updated_in_context():
                update_recordings_task = self.create_task(
                    self.update_recordings(g3, g3.recordings.events),
                    name="update_recordings",
                )
                await self.start_update_recorder_status(g3)
                try:
                    while True:
                        await self.handle_control_event(
                            g3, await self.control_events.get()
                        )
                finally:
                    await self.cancel_task(update_recordings_task)
                    await self.stop_update_recorder_status()

    async def handle_control_event(self, g3: Glasses3, event: ControlEventKind) -> None:
        logging.info(f"Handling control event: {event}")
        self.get_screen("control").set_task_running_status(True)
        match event:
            case ControlEventKind.START_RECORDING:
                await g3.recorder.start()
            case ControlEventKind.STOP_RECORDING:
                await g3.recorder.stop()
            case ControlEventKind.DELETE_RECORDING:
                await self.delete_selected_recording(g3)
            case ControlEventKind.START_LIVE:
                self.start_live_stream(g3)
            case ControlEventKind.STOP_LIVE:
                await self.stop_live_stream()
            case ControlEventKind.PLAY_RECORDING:
                await self.play_selected_recording(g3)
            case ControlEventKind.START_HEATMAP:
                self.start_heatmap()
            case ControlEventKind.STOP_HEATMAP:
                self.stop_heatmap()
        self.get_screen("control").set_task_running_status(False)

    def start_live_stream(self, g3: Glasses3) -> None:
        async def live_stream():
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                    live_screen = self.get_screen("control").ids.sm.get_screen("live")
                    Window.bind(on_resize=live_screen.clear)
                    self.latest_frame_with_timestamp = await scene_stream.get()
                    self.latest_gaze_with_timestamp = await gaze_stream.get()
                    
                    # Get actual video frame dimensions for proper alignment
                    first_frame = self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24")
                    actual_video_height, actual_video_width = first_frame.shape[:2]
                    
                    self.read_frames_task = self.create_task(
                        update_frame(scene_stream, gaze_stream, streams),
                        name="update_frame",
                    )
                    if self.live_gaze_circle is None:
                        display = live_screen.ids.display
                        video_height = display.size[0] * VIDEO_Y_TO_X_RATIO
                        video_origin_y = (display.size[1] - video_height) / 2
                        self.live_gaze_circle = GazeCircle(
                            live_screen.ids.display.canvas,
                            (0, video_origin_y),
                            (display.size[0], video_height),
                        )
                        # Initialize heatmap visualizer with ACTUAL video frame dimensions for perfect alignment
                        logging.info(f"Initializing heatmap visualizer with actual video dimensions: {actual_video_width}x{actual_video_height}")
                        logging.info(f"Display dimensions: {int(display.size[0])}x{int(video_height)}")
                        self.live_heatmap_visualizer = HeatmapVisualizer(
                            actual_video_width,
                            actual_video_height
                        )
                    self.draw_frame_event = Clock.schedule_interval(
                        draw_frame, 1 / LIVE_FRAME_RATE
                    )
                    await self.read_frames_task

        async def update_frame(scene_stream, gaze_stream, streams):
            while True:
                latest_frame_with_timestamp = await scene_stream.get()
                latest_gaze_with_timestamp = await gaze_stream.get()
                while (
                    latest_gaze_with_timestamp[1] is None
                    or latest_frame_with_timestamp[1] is None
                ):
                    if latest_frame_with_timestamp[1] is None:
                        latest_frame_with_timestamp = await scene_stream.get()
                    if latest_gaze_with_timestamp[1] is None:
                        latest_gaze_with_timestamp = await gaze_stream.get()
                while latest_gaze_with_timestamp[1] < latest_frame_with_timestamp[1]:
                    latest_gaze_with_timestamp = await gaze_stream.get()
                    while latest_gaze_with_timestamp[1] is None:
                        latest_gaze_with_timestamp = await gaze_stream.get()
                self.latest_frame_with_timestamp = latest_frame_with_timestamp
                self.latest_gaze_with_timestamp = latest_gaze_with_timestamp
                logging.debug(streams.scene_camera.stats)

        def draw_frame(dt):
            if (
                self.latest_frame_with_timestamp is None
                or self.latest_gaze_with_timestamp is None
                or self.live_gaze_circle is None
                or self.live_heatmap_visualizer is None
            ):
                logging.warning(
                    "Frame not drawn due to missing frame, gaze data, gaze circle, or heatmap visualizer."
                )
                return
            display = self.get_screen("control").ids.sm.get_screen("live").ids.display
            
            # Get original frame
            original_image = np.flip(
                self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0
            )
            
            # Update heatmap with gaze data
            gaze_data = self.latest_gaze_with_timestamp[0]
            gaze_point = None
            if len(gaze_data) != 0 and "gaze2d" in gaze_data:
                gaze_point = gaze_data["gaze2d"]
                # Update heatmap visualizer with gaze data
                self.live_heatmap_visualizer.update_gaze_data(
                    gaze_point, self.latest_gaze_with_timestamp[1]
                )
            elif len(gaze_data) != 0:
                gaze_point = gaze_data.get("gaze2d")
            
            # Apply heatmap overlay to the frame if enabled
            try:
                processed_image = self.live_heatmap_visualizer.create_heatmap_overlay(original_image)
                
                # Log performance stats occasionally
                if not hasattr(self, '_perf_counter'):
                    self._perf_counter = 0
                self._perf_counter += 1
            
                if self._perf_counter % 30 == 0:  # Every 150 frames (about every 10 seconds at 15 FPS)
                    self.live_heatmap_visualizer.reset()
                    stats = self.live_heatmap_visualizer.get_performance_stats()
                    logging.info(f"Heatmap Performance: {stats['avg_frame_time']:.1f}ms avg, "
                                f"{stats['fps_estimate']:.1f} FPS, mode: {stats['performance_mode']}, "
                                f"cache: {stats['cache_size']}, skip: {stats['skip_frames']}")
                    logging.info(f"Sliding Window: {stats['gaze_points_in_history']}/{self.live_heatmap_visualizer.max_gaze_points} points, "
                                f"counter: {stats['gaze_point_counter']}/{stats['rebuild_frequency']}")
                    logging.info(f"Heatmap dimensions: {self.live_heatmap_visualizer.width}x{self.live_heatmap_visualizer.height}")
                    logging.info(f"Video frame dimensions: {original_image.shape[1]}x{original_image.shape[0]}")
                    logging.info(f"Heatmap max value: {self.live_heatmap_visualizer.heatmap.max():.6f}")
                
                # Ensure processed image is valid
                if processed_image is None or processed_image.size == 0:
                    processed_image = original_image
                
                # Create texture from processed image with error handling
                texture = Texture.create(
                    size=(processed_image.shape[1], processed_image.shape[0]), colorfmt="bgr"
                )
                image_data = np.reshape(processed_image, -1)
                texture.blit_buffer(image_data, colorfmt="bgr", bufferfmt="ubyte")
            except Exception as e:
                # If heatmap overlay fails, fall back to original image
                logging.warning(f"Heatmap overlay failed, using original image: {e}")
                processed_image = original_image
                texture = Texture.create(
                    size=(processed_image.shape[1], processed_image.shape[0]), colorfmt="bgr"
                )
                image_data = np.reshape(processed_image, -1)
                texture.blit_buffer(image_data, colorfmt="bgr", bufferfmt="ubyte")
            
            # Update display with improved canvas management
            display.canvas.add(Color(1, 1, 1, 1))
            # Remove previous texture safely
            if self.last_texture is not None:
                try:
                    display.canvas.remove(self.last_texture)
                except ValueError:
                    # Texture not in canvas, ignore
                    pass
            
            # Add new texture to display
            self.last_texture = Rectangle(
                texture=texture,
                pos=(0, (display.top - display.width * VIDEO_Y_TO_X_RATIO) / 2),
                size=(display.width, display.width * VIDEO_Y_TO_X_RATIO),
            )
            display.canvas.add(self.last_texture)
            
            # Update gaze circle AFTER video texture is drawn to ensure it appears on top
            if gaze_point is not None:
                self.live_gaze_circle.redraw(gaze_point)

        def live_stream_task_running() -> bool:
            if self.live_stream_task is not None:
                return not self.live_stream_task.done()
            else:
                return False

        if live_stream_task_running():
            logging.info("Task not started: live_stream_task already running.")
        else:
            self.live_stream_task = self.create_task(
                live_stream(), name="live_stream_task"
            )

    async def stop_live_stream(self) -> None:
        if self.read_frames_task is not None:
            if not self.read_frames_task.cancelled():
                await self.cancel_task(self.read_frames_task)
        if self.live_stream_task is not None:
            if not self.live_stream_task.cancelled():
                await self.cancel_task(self.live_stream_task)
        if self.draw_frame_event is not None:
            self.draw_frame_event.cancel()
            self.draw_frame_event = None
        # Reset heatmap visualizer
        if self.live_heatmap_visualizer is not None:
            self.live_heatmap_visualizer.reset()
            self.live_heatmap_visualizer = None
        # Reset gaze circle
        if self.live_gaze_circle is not None:
            self.live_gaze_circle.reset()
            self.live_gaze_circle = None
        live_screen = self.get_screen("control").ids.sm.get_screen("live")
        Window.unbind(on_resize=live_screen.clear)
        live_screen.clear()
        self.last_texture = None

    def get_selected_recording(self) -> Optional[str]:
        recordings = (
            self.get_screen("control").ids.sm.get_screen("recorder").ids.recordings
        )
        selected = recordings.ids.selectables.selected_nodes
        if len(selected) != 1:
            popup = UserMessagePopup(title="No recording selected")
            popup.ids.message_label.text = "Please select a recording and try again."
            popup.open()
        else:
            return recordings.data[selected[0]]["uuid"]

    async def play_selected_recording(self, g3: Glasses3) -> None:
        uuid = self.get_selected_recording()
        if uuid is not None:
            self.get_screen("control").switch_to_screen("recording")
            recording = g3.recordings.get_recording(uuid)
            file_url = await recording.get_scenevideo_url()
            videoplayer = (
                self.get_screen("control")
                .ids.sm.get_screen("recording")
                .ids.videoplayer
            )
            videoplayer.source = file_url
            videoplayer.state = "play"

            async with aiohttp.ClientSession() as session:
                async with session.get(await recording.get_gazedata_url()) as response:
                    all_gaze_data = await response.text()
            gaze_json_list = all_gaze_data.split("\n")[:-1]
            self.gaze_data_list = []
            for gaze_json in gaze_json_list:
                self.gaze_data_list.append(json.loads(gaze_json))

            if self.replay_gaze_circle is None:
                video_height = videoplayer.size[0] * VIDEO_Y_TO_X_RATIO
                video_origin_y = (
                    videoplayer.size[1] - video_height + VIDEOPLAYER_PROGRESS_BAR_HEIGHT
                ) / 2
                self.replay_gaze_circle = GazeCircle(
                    videoplayer.canvas,
                    (0, video_origin_y),
                    (videoplayer.size[0], video_height),
                )
                self.bind_replay_gaze_updates()

    def bind_replay_gaze_updates(self):
        def reset_gaze_circle(instance, state):
            if state == "start" or state == "stop":
                if self.replay_gaze_circle is not None:
                    self.replay_gaze_circle.reset()

        def update_gaze_circle(instance, timestamp):
            if self.replay_gaze_circle is None:
                logging.warning("Gaze not drawn due to missing gaze circle.")
                return
            current_gaze_index = self.binary_search_gaze_point(
                timestamp, self.gaze_data_list
            )
            try:
                point = self.gaze_data_list[current_gaze_index]["data"]["gaze2d"]
            except KeyError:
                point = None
            self.replay_gaze_circle.redraw(point)

        videoplayer = (
            self.get_screen("control").ids.sm.get_screen("recording").ids.videoplayer
        )
        videoplayer.bind(position=update_gaze_circle)
        videoplayer.bind(state=reset_gaze_circle)

    @staticmethod
    def binary_search_gaze_point(value, gaze_list):
        left_index = 0
        right_index = len(gaze_list) - 1
        best_index = left_index
        while left_index <= right_index:
            mid_index = left_index + (right_index - left_index) // 2
            if gaze_list[mid_index]["timestamp"] < value:
                left_index = mid_index + 1
            elif gaze_list[mid_index]["timestamp"] > value:
                right_index = mid_index - 1
            else:
                best_index = mid_index
                break
            if abs(gaze_list[mid_index]["timestamp"] - value) < abs(
                gaze_list[best_index]["timestamp"] - value
            ):
                best_index = mid_index
        return best_index

    async def delete_selected_recording(self, g3: Glasses3) -> None:
        uuid = self.get_selected_recording()
        if uuid is not None:
            await g3.recordings.delete(uuid)

    async def update_recordings(self, g3, recordings_events):
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        for child in cast(List[Recording], g3.recordings):
            recorder_screen.add_recording(
                await child.get_visible_name(), child.uuid, child, atEnd=True
            )
        while True:
            event = await recordings_events.get()
            match event:
                case (RecordingsEventKind.ADDED, body):
                    uuid = cast(List[str], body)[0]
                    recording = g3.recordings.get_recording(uuid)
                    recorder_screen.add_recording(
                        await recording.get_visible_name(), recording.uuid, recording
                    )
                case (RecordingsEventKind.REMOVED, body):
                    uuid = cast(List[str], body)[0]
                    recorder_screen.remove_recording(uuid)

    async def start_update_recorder_status(self, g3: Glasses3) -> None:
        recorder_screen = self.get_screen("control").ids.sm.get_screen("recorder")
        if await g3.recorder.get_created() is not None:
            recorder_screen.set_recording_status(True)
        else:
            recorder_screen.set_recording_status(False)
        (
            recorder_started_queue,
            self.unsubscribe_to_recorder_started,
        ) = await g3.recorder.subscribe_to_started()
        (
            recorder_stopped_queue,
            self.unsubscribe_to_recorder_stopped,
        ) = await g3.recorder.subscribe_to_stopped()

        async def handle_recorder_started(
            recorder_started_queue: asyncio.Queue[SignalBody],
        ):
            while True:
                await recorder_started_queue.get()
                recorder_screen.set_recording_status(True)

        async def handle_recorder_stopped(
            recorder_stopped_queue: asyncio.Queue[SignalBody],
        ):
            while True:
                await recorder_stopped_queue.get()
                recorder_screen.set_recording_status(False)

        self.handle_recorder_started_task = self.create_task(
            handle_recorder_started(recorder_started_queue),
            name="handle_recorder_started",
        )
        self.handle_recorder_stopped_task = self.create_task(
            handle_recorder_stopped(recorder_stopped_queue),
            name="handle_recorder_stopped",
        )

    async def stop_update_recorder_status(self) -> None:
        await self.unsubscribe_to_recorder_started
        await self.unsubscribe_to_recorder_stopped
        await self.cancel_task(self.handle_recorder_started_task)
        await self.cancel_task(self.handle_recorder_stopped_task)

    def create_task(self, coro, name=None) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        logging.info(f"Task created: {task.get_name()}")
        self.tasks.add(task)
        task.add_done_callback(self.tasks.remove)
        return task

    async def cancel_task(self, task: asyncio.Task) -> None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logging.info(f"Task cancelled: {task.get_name()}")

    def start_heatmap(self):
        """Start heatmap overlay visualization."""
        if self.live_heatmap_visualizer is not None:
            self.live_heatmap_visualizer.enable_heatmap()
            logging.info("Heatmap overlay started")
        else:
            logging.warning("No heatmap visualizer available - start live feed first")
    
    def stop_heatmap(self):
        """Stop heatmap overlay visualization."""
        if self.live_heatmap_visualizer is not None:
            self.live_heatmap_visualizer.disable_heatmap()
            logging.info("Heatmap overlay stopped")
        else:
            logging.warning("No active heatmap visualizer to stop")
    
    def reset_heatmap(self):
        """Reset the heatmap visualization."""
        if self.live_heatmap_visualizer is not None:
            self.live_heatmap_visualizer.reset()
            logging.info("Heatmap reset by user")
        else:
            logging.warning("No active heatmap visualizer to reset")

    def save_heatmap(self):
        """Save the current heatmap to a file."""
        if self.live_heatmap_visualizer is not None:
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"heatmap_{timestamp}.png"
                
                # Create standalone heatmap texture
                heatmap_texture = self.live_heatmap_visualizer.create_heatmap_texture()
                if heatmap_texture is not None:
                    try:
                        # Convert texture to PIL Image and save
                        from PIL import Image
                        
                        # Get texture data
                        texture_data = heatmap_texture.get_region(0, 0, heatmap_texture.width, heatmap_texture.height)
                        # Convert RGBA bytes to PIL Image
                        img = Image.frombytes("RGBA", (heatmap_texture.width, heatmap_texture.height), texture_data)
                        img.save(filename)
                        logging.info(f"Heatmap saved as {filename}")
                        
                        # Show success popup
                        popup = UserMessagePopup(title="Heatmap Saved")
                        popup.ids.message_label.text = f"Heatmap saved as {filename}"
                        popup.open()
                    except ImportError:
                        logging.warning("PIL not available, cannot save heatmap image")
                        popup = UserMessagePopup(title="Save Failed")
                        popup.ids.message_label.text = "PIL library not available for image saving"
                        popup.open()
                else:
                    logging.warning("No heatmap data to save")
                    # Show warning popup
                    popup = UserMessagePopup(title="No Heatmap Data")
                    popup.ids.message_label.text = "No heatmap data available to save"
                    popup.open()
            except Exception as e:
                logging.error(f"Failed to save heatmap: {e}")
                # Show error popup
                popup = UserMessagePopup(title="Save Failed")
                popup.ids.message_label.text = f"Failed to save heatmap: {e}"
                popup.open()
        else:
            logging.warning("No active heatmap visualizer")
            # Show warning popup
            popup = UserMessagePopup(title="No Heatmap")
            popup.ids.message_label.text = "No active heatmap visualizer"
            popup.open()


if __name__ == "__main__":
    app = G3App()
    asyncio.run(app.async_run())
