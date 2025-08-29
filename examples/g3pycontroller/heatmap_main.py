import asyncio
import json
import logging
import math
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
            size_hint_x: 0.6
            size_hint_y: 1
        BoxLayout:
            orientation: "vertical"
            size_hint_x: 0.4
            Label:
                text: "Gaze Heatmap"
                size_hint_y: None
                height: dp(30)
            Widget:
                id: heatmap_display
                size_hint_y: 0.7
            BoxLayout:
                orientation: "horizontal"
                size_hint_y: None
                height: dp(50)
                Button:
                    text: "Start Live"
                    on_press: app.send_control_event(ControlEventKind.START_LIVE)
                Button:
                    text: "Stop Live"
                    on_press: app.send_control_event(ControlEventKind.STOP_LIVE)
            BoxLayout:
                orientation: "horizontal"
                size_hint_y: None
                height: dp(50)
                Button:
                    text: "Reset Heatmap"
                    on_press: app.reset_heatmap()
                Button:
                    text: "Save Heatmap"
                    on_press: app.save_heatmap()

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
        if atEnd == True:
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
    def __init__(self, canvas, width, height) -> None:
        self.canvas = canvas
        self.width = width
        self.height = height
        
        # Heatmap configuration
        self.sigma = 20.0  # Reduced for performance
        self.decay_rate = 0.98
        self.max_intensity = 1.0
        self.blend_alpha = 0.7
        
        # Performance optimization settings
        self.update_counter = 0
        self.update_frequency = 3  # Update display every 3 frames
        self.texture_cache_time = 0
        self.texture_cache_duration = 0.1  # Cache texture for 100ms
        
        # Initialize heatmap data
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.last_update_time = time.time()
        self.gaze_history = deque(maxlen=50)  # Reduced for performance
        
        # Initialize colormap for heatmap visualization
        self._initialize_colormap()
        
        # Initialize graphics objects
        self.heatmap_texture = None
        self.heatmap_rect = None
        self.cached_texture = None
        
        logging.info(f"HeatmapVisualizer initialized: {width}x{height}")
    
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
    
    def update_gaze_data(self, gaze_point, timestamp):
        """Update heatmap with new gaze data."""
        if gaze_point is None or len(gaze_point) != 2:
            return
        
        # Convert normalized coordinates to heatmap pixel coordinates
        # gaze_point is normalized (0-1), so scale to heatmap dimensions
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
    
    def _update_heatmap(self, x, y, timestamp):
        """Update the heatmap with temporal decay - optimized version."""
        current_time = time.time()
        dt = current_time - self.last_update_time
        
        # Apply temporal decay only every few updates for performance
        if dt > 0.1:  # Decay every 100ms
            decay_factor = math.exp(-dt / self.decay_rate)
            self.heatmap *= decay_factor
            self.last_update_time = current_time
        
        # Add new fixation point with optimized Gaussian blob
        if 0 <= x < self.width and 0 <= y < self.height:
            size = int(self.sigma * 1.5)  # Reduced coverage for performance
            
            # Calculate bounds for the Gaussian
            y1, y2 = max(0, y - size), min(self.height, y + size + 1)
            x1, x2 = max(0, x - size), min(self.width, x + size + 1)
            
            if x2 > x1 and y2 > y1:
                # Use simpler, faster Gaussian approximation
                w, h = x2 - x1, y2 - y1
                cy, cx = y - y1, x - x1
                
                # Fast Gaussian using broadcasting
                yy = np.arange(h)[:, None]
                xx = np.arange(w)[None, :]
                
                # Simplified Gaussian calculation
                dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
                gaussian = np.exp(-dist_sq / (self.sigma ** 2))
                
                # Add to heatmap
                self.heatmap[y1:y2, x1:x2] += gaussian * 0.5  # Reduced intensity
        
        # Normalize less frequently for performance
        if current_time - self.last_update_time > 0.5:  # Every 500ms
            if self.heatmap.max() > self.max_intensity:
                self.heatmap = (self.heatmap / self.heatmap.max()) * self.max_intensity
    
    def create_heatmap_texture(self):
        """Create Kivy texture from heatmap data with caching."""
        current_time = time.time()
        
        # Check if we can use cached texture
        if (self.cached_texture is not None and 
            current_time - self.texture_cache_time < self.texture_cache_duration):
            return self.cached_texture
            
        if self.heatmap.max() == 0:
            return None
        
        # Skip smoothing for performance - use raw heatmap
        # smoothed_heatmap = self._apply_simple_smoothing(self.heatmap)
        
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
        
        # Cache the texture
        self.cached_texture = texture
        self.texture_cache_time = current_time
        
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
    
    def update_display(self, display_pos, display_size):
        """Update heatmap display on canvas with frequency control."""
        # Update counter for frequency control
        self.update_counter += 1
        
        # Only update display every N frames for performance
        if self.update_counter % self.update_frequency != 0:
            return
        
        # Remove previous heatmap safely
        if self.heatmap_rect is not None:
            try:
                self.canvas.remove(self.heatmap_rect)
            except ValueError:
                # Rectangle not in canvas, ignore
                pass
            self.heatmap_rect = None
        
        # Create new heatmap texture
        texture = self.create_heatmap_texture()
        if texture is None:
            return
        
        # Debug: Log canvas and positioning info occasionally
        if self.update_counter % 300 == 0:  # Every 100 updates (since we update every 3 frames)
            logging.debug(f"HeatmapVisualizer: Drawing on canvas {id(self.canvas)}")
            logging.debug(f"HeatmapVisualizer: Position={display_pos}, Size={display_size}")
            logging.debug(f"HeatmapVisualizer: Heatmap dimensions={self.width}x{self.height}")
        
        # Add heatmap to canvas - this should be the heatmap_display canvas (RIGHT side)
        self.heatmap_texture = texture
        self.heatmap_rect = Rectangle(
            texture=texture,
            pos=display_pos,
            size=display_size
        )
        self.canvas.add(self.heatmap_rect)
    
    def reset(self):
        """Reset heatmap data."""
        self.heatmap.fill(0)
        self.gaze_history.clear()
        self.last_update_time = time.time()
        
        # Reset performance counters and caches
        self.update_counter = 0
        self.texture_cache_time = 0
        self.cached_texture = None
        
        # Remove heatmap from canvas safely
        if self.heatmap_rect is not None:
            try:
                self.canvas.remove(self.heatmap_rect)
            except ValueError:
                # Rectangle not in canvas, ignore
                pass
            self.heatmap_rect = None
        
        logging.info("HeatmapVisualizer reset")


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
        self.get_screen("control").set_task_running_status(False)

    def start_live_stream(self, g3: Glasses3) -> None:
        async def live_stream():
            async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
                async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                    live_screen = self.get_screen("control").ids.sm.get_screen("live")
                    Window.bind(on_resize=live_screen.clear)
                    self.latest_frame_with_timestamp = await scene_stream.get()
                    self.latest_gaze_with_timestamp = await gaze_stream.get()
                    self.read_frames_task = self.create_task(
                        update_frame(scene_stream, gaze_stream, streams),
                        name="update_frame",
                    )
                    if self.live_gaze_circle is None:
                        display = live_screen.ids.display
                        heatmap_display = live_screen.ids.heatmap_display
                        video_height = display.size[0] * VIDEO_Y_TO_X_RATIO
                        video_origin_y = (display.size[1] - video_height) / 2
                        self.live_gaze_circle = GazeCircle(
                            live_screen.ids.display.canvas,
                            (0, video_origin_y),
                            (display.size[0], video_height),
                        )
                        # Initialize heatmap visualizer with separate canvas
                        # Ensure we have valid dimensions for the heatmap display
                        heatmap_width = max(int(heatmap_display.size[0]), 100)
                        heatmap_height = max(int(heatmap_display.size[1]), 100)
                        logging.info(f"Initializing heatmap visualizer on RIGHT side widget with dimensions: {heatmap_width}x{heatmap_height}")
                        logging.info(f"Heatmap display position: {heatmap_display.pos}, size: {heatmap_display.size}")
                        self.live_heatmap_visualizer = HeatmapVisualizer(
                            heatmap_display.canvas,
                            heatmap_width,
                            heatmap_height
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
            image = np.flip(
                self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24"), 0
            )
            texture = Texture.create(
                size=(image.shape[1], image.shape[0]), colorfmt="bgr"
            )
            image = np.reshape(image, -1)
            texture.blit_buffer(image, colorfmt="bgr", bufferfmt="ubyte")
            display.canvas.add(Color(1, 1, 1, 1))
            # Remove previous texture safely
            if self.last_texture is not None:
                try:
                    display.canvas.remove(self.last_texture)
                except ValueError:
                    # Texture not in canvas, ignore
                    pass
            self.last_texture = Rectangle(
                texture=texture,
                pos=(0, (display.top - display.width * VIDEO_Y_TO_X_RATIO) / 2),
                size=(display.width, display.width * VIDEO_Y_TO_X_RATIO),
            )
            display.canvas.add(self.last_texture)
            
            # Update and display heatmap in separate widget (RIGHT side)
            gaze_data = self.latest_gaze_with_timestamp[0]
            if len(gaze_data) != 0 and "gaze2d" in gaze_data:
                point = gaze_data["gaze2d"]
                # Update heatmap with gaze data
                self.live_heatmap_visualizer.update_gaze_data(
                    point, self.latest_gaze_with_timestamp[1]
                )
                # Get heatmap display widget and update - ENSURE this is the RIGHT side widget
                heatmap_display = self.get_screen("control").ids.sm.get_screen("live").ids.heatmap_display
                
                # Debug: Log heatmap positioning details occasionally
                if not hasattr(self, '_heatmap_debug_counter'):
                    self._heatmap_debug_counter = 0
                self._heatmap_debug_counter += 1
                
                if self._heatmap_debug_counter % 100 == 0:  # Log every 100 frames
                    logging.debug(f"Heatmap widget pos: {heatmap_display.pos}, size: {heatmap_display.size}")
                    logging.debug(f"Gaze point: {point}")
                
                # Check if heatmap widget has valid size, if not skip this frame
                if heatmap_display.size[0] > 0 and heatmap_display.size[1] > 0:
                    # Position heatmap at (0,0) relative to the heatmap_display widget
                    self.live_heatmap_visualizer.update_display(
                        display_pos=(0, 0),  # Position within the heatmap_display widget
                        display_size=(heatmap_display.size[0], heatmap_display.size[1])
                    )
                else:
                    # Widget not yet properly sized, skip this frame
                    if self._heatmap_debug_counter % 100 == 0:
                        logging.warning(f"Heatmap widget not yet sized properly: {heatmap_display.size}")
                # Also update the gaze circle for immediate feedback on the LEFT side video
                self.live_gaze_circle.redraw(point)
            else:
                # Still update gaze circle even if no gaze data for heatmap
                if len(gaze_data) != 0:
                    point = gaze_data.get("gaze2d")
                    self.live_gaze_circle.redraw(point)

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
        # Reset debug counter
        if hasattr(self, '_heatmap_debug_counter'):
            delattr(self, '_heatmap_debug_counter')
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
        if await g3.recorder.get_created() != None:
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
                
                # Create heatmap image
                heatmap_texture = self.live_heatmap_visualizer.create_heatmap_texture()
                if heatmap_texture is not None:
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
