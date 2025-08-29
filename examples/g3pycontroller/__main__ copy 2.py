import asyncio
import json
import logging
import time
from datetime import datetime

from typing import List, Optional, Set, Tuple, cast

import aiohttp
import cv2
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

from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider


from g3pylib import Glasses3, connect_to_glasses
from g3pylib.g3typing import SignalBody
from g3pylib.recordings import RecordingsEventKind
from g3pylib.recordings.recording import Recording
from g3pylib.zeroconf import EventKind, G3Service, G3ServiceDiscovery

from heatmap_visualizer import HeatmapVisualizer
from intent_predictor import IntentPredictor

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
            Label:
                text: "Intent Prediction"
                size_hint_y: None
                height: dp(30)
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(180)
                Button:
                    id: intent_start_btn
                    text: "Start Intent"
                    on_press: app.send_control_event(ControlEventKind.START_INTENT_PREDICTION)
                Button:
                    id: intent_stop_btn
                    text: "Stop Intent"
                    on_press: app.send_control_event(ControlEventKind.STOP_INTENT_PREDICTION)
                Button:
                    text: "Configure"
                    on_press: app.configure_intent_prediction()
                Label:
                    id: intent_status
                    text: "Intent: Not active"
                    size_hint_y: None
                    height: dp(30)
                    text_size: self.size
                    halign: "center"
                    valign: "middle"
                Label:
                    id: intent_timing
                    text: "Timing: --"
                    size_hint_y: None
                    height: dp(30)
                    text_size: self.size
                    halign: "center"
                    valign: "middle"
                    color: (0.7, 0.7, 0.7, 1)
            Label:
                text: "Image Saving"
                size_hint_y: None
                height: dp(30)
            BoxLayout:
                orientation: "vertical"
                size_hint_y: None
                height: dp(150)
                Button:
                    id: image_save_start_btn
                    text: "Start Saving"
                    on_press: app.start_image_saving()
                Button:
                    id: image_save_stop_btn
                    text: "Stop Saving"
                    on_press: app.stop_image_saving()
                Button:
                    text: "Configure"
                    on_press: app.configure_image_saving()
                Label:
                    id: image_save_status
                    text: "Saving: Disabled"
                    size_hint_y: None
                    height: dp(30)
                    text_size: self.size
                    halign: "center"
                    valign: "middle"
                    color: (0.7, 0.7, 0.7, 1)
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
        _ = rv, index  # Unused parameters
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
        _ = args  # Unused parameter
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


def draw_text_on_frame(frame: np.ndarray, text: str, position: tuple, 
                      font_scale: float = 1.0, color: tuple = (255, 255, 255), 
                      thickness: int = 2, background_color: tuple = None,
                      background_padding: int = 10, is_flipped: bool = False) -> np.ndarray:
    """
    Draw text on a frame with optional background.
    
    Args:
        frame: Input frame (BGR format)
        text: Text to draw
        position: (x, y) position for text (top-left origin)
        font_scale: Font size scaling factor
        color: Text color in BGR format
        thickness: Text thickness
        background_color: Optional background color in BGR format
        background_padding: Padding around text for background
        is_flipped: True if frame is already vertically flipped, False if not
    
    Returns:
        Frame with text drawn
    """
    try:
        # Make a copy to avoid modifying original
        frame_copy = frame.copy()
        
        # Handle multi-line text
        lines = text.split('\n')
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_height = int(30 * font_scale)  # Approximate line height
        
        # Get text size for the longest line for background
        max_text_width = 0
        for line in lines:
            if line.strip():
                (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
                max_text_width = max(max_text_width, text_width)
        
        frame_height = frame_copy.shape[0]
        x, y = position
        
        if is_flipped:
            # For a vertically flipped frame, adjust coordinates
            total_text_height = len(lines) * line_height
            adjusted_y = frame_height - y - total_text_height + line_height
            adjusted_position = (x, adjusted_y)
        else:
            # For normal (unflipped) frame, use coordinates directly
            adjusted_position = (x, y + line_height)  # Add line_height for baseline offset
        
        # Draw background if specified
        if background_color is not None and lines and max_text_width > 0:
            if is_flipped:
                bg_x1 = adjusted_position[0] - background_padding
                bg_y1 = adjusted_position[1] - line_height - background_padding
                bg_x2 = adjusted_position[0] + max_text_width + background_padding
                bg_y2 = adjusted_position[1] + (len(lines) - 1) * line_height + background_padding
            else:
                bg_x1 = x - background_padding
                bg_y1 = y - background_padding
                bg_x2 = x + max_text_width + background_padding
                bg_y2 = y + len(lines) * line_height + background_padding
            
            # Ensure background rectangle is within frame bounds
            bg_x1 = max(0, bg_x1)
            bg_y1 = max(0, bg_y1)
            bg_x2 = min(frame_copy.shape[1], bg_x2)
            bg_y2 = min(frame_copy.shape[0], bg_y2)
            
            cv2.rectangle(frame_copy, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
        
        # Draw each line of text
        for i, line in enumerate(lines):
            if line.strip():  # Only draw non-empty lines
                line_position = (adjusted_position[0], adjusted_position[1] + i * line_height)
                cv2.putText(frame_copy, line, line_position, font, font_scale, color, thickness)
        
        return frame_copy
        
    except Exception as e:
        logging.error(f"Failed to draw text on frame: {e}")
        return frame  # Return original frame on error


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
        self.intent_predictor = IntentPredictor()
        self.intent_update_event = None
        self.intent_prediction_task: Optional[asyncio.Task] = None

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
            case ControlEventKind.START_INTENT_PREDICTION:
                self.start_intent_prediction()
            case ControlEventKind.STOP_INTENT_PREDICTION:
                self.stop_intent_prediction()
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
                        # Set to balanced mode for optimal performance
                        self.live_heatmap_visualizer.set_performance_mode("balanced")
                    self.draw_frame_event = Clock.schedule_interval(
                        draw_frame, 1 / LIVE_FRAME_RATE
                    )
                    # Start async intent prediction task
                    self.start_intent_prediction_task()
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
            _ = dt  # Unused parameter
            if (
                self.latest_frame_with_timestamp is None
                or self.latest_gaze_with_timestamp is None
                or self.live_gaze_circle is None
                or self.live_heatmap_visualizer is None
            ):
                logging.warning("Frame not drawn due to missing data")
                return
            display = self.get_screen("control").ids.sm.get_screen("live").ids.display
            
            # Get original frame WITHOUT flipping first - we'll flip at the end
            original_image_unflipped = self.latest_frame_with_timestamp[0].to_ndarray(format="bgr24")
            
            gaze_data = self.latest_gaze_with_timestamp[0]
            gaze_point = None
            if len(gaze_data) != 0 and "gaze2d" in gaze_data:
                gaze_point = gaze_data["gaze2d"]
                # Update heatmap with gaze data (reduce logging frequency for performance)
                self.live_heatmap_visualizer.update_gaze_data(gaze_point, self.latest_gaze_with_timestamp[1])
            
            # Create flipped version for heatmap processing (heatmap visualizer expects flipped frames)
            original_image_flipped = np.flip(original_image_unflipped, 0)
            
            try:
                # Create heatmap overlay on flipped frame
                processed_image = self.live_heatmap_visualizer.create_heatmap_overlay(original_image_flipped)
                if processed_image is None or processed_image.size == 0:
                    processed_image = original_image_flipped
                
                # Update intent predictor with frames (use flipped versions for consistency)
                self.intent_predictor.update_frames(original_image_flipped, processed_image)
                
                # Add intent prediction text overlay on the UNFLIPPED processed frame
                # First, create unflipped version of processed image for text overlay
                processed_image_for_text = np.flip(processed_image, 0)
                
                latest_prediction = self.intent_predictor.get_latest_prediction_safe()
                if latest_prediction and self.intent_predictor.is_enabled:
                    try:
                        # Format prediction text for overlay
                        intent_text = latest_prediction.get('prediction', 'unknown')
                        reasoning = latest_prediction.get('reasoning', '')
                        timing_ms = latest_prediction.get('duration_ms', 0)
                        is_executing = latest_prediction.get('is_executing', False)
                        
                        # Create display text
                        if is_executing:
                            display_text = f"{intent_text}\nProcessing..."
                        else:
                            first_line = intent_text
                            if timing_ms and timing_ms > 0:
                                first_line += f" ({timing_ms:.0f}ms)"
                            
                            # Truncate reasoning for display
                            second_line = ""
                            if reasoning:
                                max_reasoning_length = 45
                                if len(reasoning) > max_reasoning_length:
                                    second_line = f"{reasoning[:max_reasoning_length]}..."
                                else:
                                    second_line = reasoning
                            
                            display_text = f"{first_line}\n{second_line}" if second_line else first_line
                        
                        # Determine text color based on timing/state
                        if is_executing:
                            bg_color = (0, 128, 255)  # Orange in BGR
                        elif timing_ms and timing_ms > 0:
                            if timing_ms < 1000:
                                bg_color = (0, 200, 0)  # Green in BGR
                            elif timing_ms < 3000:
                                bg_color = (0, 255, 255)  # Yellow in BGR
                            else:
                                bg_color = (0, 0, 200)  # Red in BGR
                        else:
                            bg_color = (80, 80, 80)  # Gray in BGR
                        
                        # Draw text on UNFLIPPED processed image (is_flipped=False)
                        text_position = (20, 50)  # Top-left corner with margins
                        processed_image_for_text = draw_text_on_frame(
                            processed_image_for_text, 
                            display_text, 
                            text_position,
                            font_scale=0.8,
                            color=(255, 255, 255),  # White text
                            thickness=2,
                            background_color=bg_color,
                            background_padding=8,
                            is_flipped=False  # Frame is NOT flipped at this point
                        )
                        
                    except Exception as e:
                        logging.error(f"Failed to add intent text overlay: {e}")
                
                # Now flip the frame with text overlay back for display
                final_image = np.flip(processed_image_for_text, 0)
                
                texture = Texture.create(
                    size=(final_image.shape[1], final_image.shape[0]), colorfmt="bgr"
                )
                image_data = np.reshape(final_image, -1)
                texture.blit_buffer(image_data, colorfmt="bgr", bufferfmt="ubyte")
            except Exception as e:
                logging.error("Texture update failed: %s", e)
                # Fallback mode - apply text to unflipped original frame
                processed_image_for_text = original_image_unflipped.copy()
                
                # Add intent prediction text overlay even in fallback mode
                latest_prediction = self.intent_predictor.get_latest_prediction_safe()
                if latest_prediction and self.intent_predictor.is_enabled:
                    try:
                        # Format prediction text for overlay
                        intent_text = latest_prediction.get('prediction', 'unknown')
                        reasoning = latest_prediction.get('reasoning', '')
                        timing_ms = latest_prediction.get('duration_ms', 0)
                        is_executing = latest_prediction.get('is_executing', False)
                        
                        # Create display text
                        if is_executing:
                            display_text = f"{intent_text}\nProcessing..."
                        else:
                            first_line = intent_text
                            if timing_ms and timing_ms > 0:
                                first_line += f" ({timing_ms:.0f}ms)"
                            
                            # Truncate reasoning for display
                            second_line = ""
                            if reasoning:
                                max_reasoning_length = 45
                                if len(reasoning) > max_reasoning_length:
                                    second_line = f"{reasoning[:max_reasoning_length]}..."
                                else:
                                    second_line = reasoning
                            
                            display_text = f"{first_line}\n{second_line}" if second_line else first_line
                        
                        # Determine text color based on timing/state
                        if is_executing:
                            bg_color = (0, 128, 255)  # Orange in BGR
                        elif timing_ms and timing_ms > 0:
                            if timing_ms < 1000:
                                bg_color = (0, 200, 0)  # Green in BGR
                            elif timing_ms < 3000:
                                bg_color = (0, 255, 255)  # Yellow in BGR
                            else:
                                bg_color = (0, 0, 200)  # Red in BGR
                        else:
                            bg_color = (80, 80, 80)  # Gray in BGR
                        
                        # Draw text on UNFLIPPED processed image (is_flipped=False)
                        text_position = (20, 50)  # Top-left corner with margins
                        processed_image_for_text = draw_text_on_frame(
                            processed_image_for_text, 
                            display_text, 
                            text_position,
                            font_scale=0.8,
                            color=(255, 255, 255),  # White text
                            thickness=2,
                            background_color=bg_color,
                            background_padding=8,
                            is_flipped=False  # Frame is NOT flipped at this point
                        )
                        
                    except Exception as text_e:
                        logging.error(f"Failed to add intent text overlay in fallback: {text_e}")
                
                # Flip the frame for display
                final_image = np.flip(processed_image_for_text, 0)
                
                texture = Texture.create(
                    size=(final_image.shape[1], final_image.shape[0]), colorfmt="bgr"
                )
                image_data = np.reshape(final_image, -1)
                texture.blit_buffer(image_data, colorfmt="bgr", bufferfmt="ubyte")
                
            display.canvas.add(Color(1, 1, 1, 1))
            if self.last_texture is not None:
                try:
                    display.canvas.remove(self.last_texture)
                except ValueError:
                    pass
            self.last_texture = Rectangle(
                texture=texture,
                pos=(0, (display.top - display.width * VIDEO_Y_TO_X_RATIO) / 2),
                size=(display.width, display.width * VIDEO_Y_TO_X_RATIO),
            )
            display.canvas.add(self.last_texture)
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
        if self.intent_update_event is not None:
            self.intent_update_event.cancel()
            self.intent_update_event = None
        # Stop intent prediction task
        await self.stop_intent_prediction_task()
        # Reset heatmap visualizer
        if self.live_heatmap_visualizer is not None and not self.live_heatmap_visualizer.is_enabled:
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
    
    def set_heatmap_performance_mode(self, mode="balanced"):
        """Set heatmap performance mode: fast, balanced, or quality."""
        if self.live_heatmap_visualizer is not None:
            self.live_heatmap_visualizer.set_performance_mode(mode)
            logging.info(f"Heatmap performance mode set to: {mode}")
        else:
            logging.warning("No active heatmap visualizer")
    
    def get_heatmap_performance_stats(self):
        """Get heatmap performance statistics."""
        if self.live_heatmap_visualizer is not None:
            stats = self.live_heatmap_visualizer.get_performance_stats()
            # Add additional app-level stats
            stats.update({
                'live_frame_rate': LIVE_FRAME_RATE,
                'is_live_streaming': self.live_stream_task is not None and not self.live_stream_task.done(),
                'intent_prediction_active': self.intent_predictor.is_enabled if hasattr(self, 'intent_predictor') else False
            })
            return stats
        else:
            return None
    
    def log_performance_stats(self):
        """Log current performance statistics."""
        stats = self.get_heatmap_performance_stats()
        if stats:
            logging.info(f"Heatmap Performance - Update time: {stats['avg_update_time_ms']:.2f}ms, "
                        f"Kernel: {stats['kernel_size']}x{stats['kernel_size']}, "
                        f"Update freq: every {stats['update_frequency']} frames, "
                        f"Pre-computed: {stats['has_precomputed_kernel']}")
        else:
            logging.info("No heatmap performance stats available")

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
    
    def start_intent_prediction(self):
        """Start real-time intent prediction."""
        try:
            # Check if we need an API key
            if not self.intent_predictor.client:
                # Show popup to get API key
                self.show_api_key_popup()
                return
            
            if self.intent_predictor.enable_prediction():
                # Start background task if live stream is running
                if (self.live_stream_task is not None and 
                    not self.live_stream_task.done() and 
                    (self.intent_prediction_task is None or self.intent_prediction_task.done())):
                    self.start_intent_prediction_task()
                
                # Update UI
                live_screen = self.get_screen("control").ids.sm.get_screen("live")
                live_screen.ids.intent_status.text = "Intent: Active"
                live_screen.ids.intent_timing.text = "Timing: Waiting..."
                live_screen.ids.intent_timing.color = (0.7, 0.7, 0.7, 1)  # Gray
                
                
                
                logging.info("Intent prediction started")
            else:
                logging.warning("Failed to start intent prediction")
        except Exception as e:
            logging.error(f"Error starting intent prediction: {e}")
    
    def stop_intent_prediction(self):
        """Stop real-time intent prediction."""
        self.intent_predictor.disable_prediction()
        # Update UI
        live_screen = self.get_screen("control").ids.sm.get_screen("live")
        live_screen.ids.intent_status.text = "Intent: Not active"
        live_screen.ids.intent_timing.text = "Timing: --"
        live_screen.ids.intent_timing.color = (0.7, 0.7, 0.7, 1)  # Gray
        

        
        logging.info("Intent prediction stopped")
    
    def update_intent_prediction(self, dt):
        """Legacy method - now handled by async background task."""
        _ = dt  # Unused parameter
        # This method is kept for backward compatibility but no longer used
        pass
    
    def start_intent_prediction_task(self):
        """Start the background intent prediction task."""
        if self.intent_prediction_task is None or self.intent_prediction_task.done():
            self.intent_prediction_task = self.create_task(
                self.intent_prediction_background_loop(), 
                name="intent_prediction_background"
            )
            logging.info("Intent prediction background task started")
    
    async def stop_intent_prediction_task(self):
        """Stop the background intent prediction task."""
        if self.intent_prediction_task is not None and not self.intent_prediction_task.done():
            await self.cancel_task(self.intent_prediction_task)
            self.intent_prediction_task = None
            logging.info("Intent prediction background task stopped")
    
    async def intent_prediction_background_loop(self):
        """Background loop for intent prediction."""
        while True:
            try:
                # Check if intent prediction is enabled
                if self.intent_predictor.is_enabled:
                    # Set executing state in the prediction for overlay display
                    executing_prediction = {
                        "prediction": "Processing...", 
                        "reasoning": "", 
                        "is_executing": True,
                        "timestamp": time.time(),
                        "duration_ms": 0
                    }
                    
                    # Store executing state in predictor
                    with self.intent_predictor._prediction_lock:
                        self.intent_predictor._latest_prediction = executing_prediction
                    
                    # Show that prediction is starting in UI
                    Clock.schedule_once(
                        lambda dt: self.update_intent_ui_executing(), 0
                    )
                    
                    # Run prediction in background
                    prediction = await self.intent_predictor.predict_intent_async()
                    
                    if prediction:
                        # Mark as not executing and schedule UI update
                        prediction['is_executing'] = False
                        Clock.schedule_once(
                            lambda dt: self.update_intent_ui(prediction), 0
                        )
                
                # Wait for the configured interval
                await asyncio.sleep(self.intent_predictor.prediction_interval)
                
            except asyncio.CancelledError:
                logging.info("Intent prediction background loop cancelled")
                break
            except Exception as e:
                logging.error(f"Error in intent prediction background loop: {e}")
                # Continue the loop even on errors
                await asyncio.sleep(1.0)
    
    def update_intent_ui_executing(self):
        """Update the UI to show that intent prediction is executing (runs on main thread)."""
        try:
            live_screen = self.get_screen("control").ids.sm.get_screen("live")
            live_screen.ids.intent_status.text = "Intent: Executing..."
            live_screen.ids.intent_timing.text = "Timing: Processing..."
            live_screen.ids.intent_timing.color = (1, 0.8, 0, 1)  # Orange color for executing
            

            
            logging.debug("Intent UI updated: Executing")
        except Exception as e:
            logging.error(f"Error updating intent UI (executing): {e}")
    
    def update_intent_ui(self, prediction: dict):
        """Update the UI with the latest intent prediction (runs on main thread)."""
        try:
            live_screen = self.get_screen("control").ids.sm.get_screen("live")
            intent_text = f"Intent: {prediction['prediction']}"
            live_screen.ids.intent_status.text = intent_text
            
            # Update timing information
            timing_ms = None
            if 'duration_ms' in prediction and prediction['duration_ms'] > 0:
                timing_ms = prediction['duration_ms']
                timing_text = f"Timing: {timing_ms}ms"
                live_screen.ids.intent_timing.text = timing_text
                # Color coding based on timing - green for fast, yellow for medium, red for slow
                if timing_ms < 1000:  # Less than 1 second
                    live_screen.ids.intent_timing.color = (0, 1, 0, 1)  # Green
                elif timing_ms < 3000:  # Less than 3 seconds
                    live_screen.ids.intent_timing.color = (1, 1, 0, 1)  # Yellow
                else:  # 3+ seconds
                    live_screen.ids.intent_timing.color = (1, 0, 0, 1)  # Red
            else:
                live_screen.ids.intent_timing.text = "Timing: --"
                live_screen.ids.intent_timing.color = (0.7, 0.7, 0.7, 1)  # Gray
            

            
            logging.debug(f"Intent UI updated: {prediction['prediction']} (took {prediction.get('duration_ms', 0)}ms)")
        except Exception as e:
            logging.error(f"Error updating intent UI: {e}")
    
    def show_api_key_popup(self):
        """Show popup to enter OpenAI API key."""
        from kivy.uix.textinput import TextInput
        from kivy.uix.button import Button
        from kivy.uix.boxlayout import BoxLayout
        
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        content.add_widget(Label(
            text="Enter your OpenAI API key for intent prediction:",
            size_hint_y=None,
            height=dp(30)
        ))
        
        api_key_input = TextInput(
            hint_text="sk-...",
            password=True,
            multiline=False,
            size_hint_y=None,
            height=dp(40)
        )
        content.add_widget(api_key_input)
        
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50))
        
        def on_confirm(*args):
            api_key = api_key_input.text.strip()
            if api_key:
                self.intent_predictor.set_api_key(api_key)
                popup.dismiss()
                # Try to start prediction again
                self.start_intent_prediction()
            else:
                api_key_input.hint_text = "Please enter a valid API key"
        
        def on_cancel(*args):
            popup.dismiss()
        
        confirm_btn = Button(text="Confirm")
        confirm_btn.bind(on_press=on_confirm)
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_press=on_cancel)
        
        button_layout.add_widget(cancel_btn)
        button_layout.add_widget(confirm_btn)
        content.add_widget(button_layout)
        
        popup = Popup(
            title="OpenAI API Key Required",
            content=content,
            size_hint=(0.8, 0.6)
        )
        popup.open()
    
    def configure_intent_prediction(self):
        """Show configuration popup for intent prediction."""

        
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Prediction interval slider
        content.add_widget(Label(
            text=f"Prediction Interval: {self.intent_predictor.prediction_interval:.1f}s",
            size_hint_y=None,
            height=dp(30)
        ))
        
        interval_slider = Slider(
            min=0.5, max=10.0, 
            value=self.intent_predictor.prediction_interval,
            step=0.5,
            size_hint_y=None,
            height=dp(40)
        )
        
        interval_label = content.children[0]
        
        def on_interval_change(instance, value):
            interval_label.text = f"Prediction Interval: {value:.1f}s"
        
        interval_slider.bind(value=on_interval_change)
        content.add_widget(interval_slider)
        
        # Candidates text input
        content.add_widget(Label(
            text="Candidate Actions (one per line):",
            size_hint_y=None,
            height=dp(30)
        ))
        
        candidates_input = TextInput(
            text="\n".join(self.intent_predictor.current_candidates),
            multiline=True,
            size_hint_y=0.6
        )
        content.add_widget(candidates_input)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50))
        
        def on_save(*args):
            # Update interval
            self.intent_predictor.set_prediction_interval(interval_slider.value)
            
            # Update candidates
            candidates_text = candidates_input.text.strip()
            if candidates_text:
                candidates = [line.strip() for line in candidates_text.split('\n') if line.strip()]
                self.intent_predictor.set_candidates(candidates)
            
            popup.dismiss()
        
        def on_cancel(*args):
            popup.dismiss()
        
        save_btn = Button(text="Save")
        save_btn.bind(on_press=on_save)
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_press=on_cancel)
        
        button_layout.add_widget(cancel_btn)
        button_layout.add_widget(save_btn)
        content.add_widget(button_layout)
        
        popup = Popup(
            title="Intent Prediction Configuration",
            content=content,
            size_hint=(0.9, 0.8)
        )
        popup.open()
    
    def start_image_saving(self):
        """Start saving images to disk."""
        try:
            # Check if intent predictor is available
            if not hasattr(self, 'intent_predictor') or self.intent_predictor is None:
                popup = UserMessagePopup(title="Image Saving Not Available")
                popup.ids.message_label.text = "Intent predictor not initialized. Start live feed first."
                popup.open()
                return
            
            # Enable image saving with default directory
            from datetime import datetime as dt
            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"saved_frames_{timestamp}"
            self.intent_predictor.enable_image_saving(save_dir)
            
            # Update UI
            live_screen = self.get_screen("control").ids.sm.get_screen("live")
            live_screen.ids.image_save_status.text = "Saving: Active"
            live_screen.ids.image_save_status.color = (0, 1, 0, 1)  # Green
            
            logging.info(f"Image saving started - directory: {save_dir}")
            
            # Show success popup
            popup = UserMessagePopup(title="Image Saving Started")
            popup.ids.message_label.text = f"Scene & heatmap frames will be saved to: {save_dir}"
            popup.open()
            
        except Exception as e:
            logging.error(f"Failed to start image saving: {e}")
            popup = UserMessagePopup(title="Error")
            popup.ids.message_label.text = f"Failed to start image saving: {e}"
            popup.open()
    
    def stop_image_saving(self):
        """Stop saving images to disk."""
        try:
            if hasattr(self, 'intent_predictor') and self.intent_predictor is not None:
                stats = self.intent_predictor.get_image_saving_stats()
                self.intent_predictor.disable_image_saving()
                
                # Update UI
                live_screen = self.get_screen("control").ids.sm.get_screen("live")
                live_screen.ids.image_save_status.text = "Saving: Disabled"
                live_screen.ids.image_save_status.color = (0.7, 0.7, 0.7, 1)  # Gray
                
                logging.info("Image saving stopped")
                
                # Show summary popup
                popup = UserMessagePopup(title="Image Saving Stopped")
                total_predictions = stats.get('total_predictions', stats['saved_frame_counter'])
                popup.ids.message_label.text = f"Saved {total_predictions} predictions with scene & heatmap frames"
                popup.open()
            else:
                popup = UserMessagePopup(title="No Active Saving")
                popup.ids.message_label.text = "Image saving was not active"
                popup.open()
                
        except Exception as e:
            logging.error(f"Failed to stop image saving: {e}")
    
    def configure_image_saving(self):
        """Show configuration popup for image saving."""
        if not hasattr(self, 'intent_predictor') or self.intent_predictor is None:
            popup = UserMessagePopup(title="Image Saving Not Available")
            popup.ids.message_label.text = "Intent predictor not initialized. Start live feed first."
            popup.open()
            return
        
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        
        # Current status
        stats = self.intent_predictor.get_image_saving_stats()
        content.add_widget(Label(
            text=f"Status: {'Enabled' if stats['save_images_enabled'] else 'Disabled'} (Scene + heatmap only)",
            size_hint_y=None,
            height=dp(30)
        ))
        
        if stats['save_images_enabled']:
            content.add_widget(Label(
                text=f"Directory: {stats['save_directory']}",
                size_hint_y=None,
                height=dp(30)
            ))
            content.add_widget(Label(
                text=f"Frames saved: {stats['saved_frame_counter']}",
                size_hint_y=None,
                height=dp(30)
            ))
        
        # Directory input
        content.add_widget(Label(
            text="Save Directory Name:",
            size_hint_y=None,
            height=dp(30)
        ))
        
        directory_input = TextInput(
            text="saved_frames",
            multiline=False,
            size_hint_y=None,
            height=dp(40)
        )
        content.add_widget(directory_input)
        
        # Format selection
        content.add_widget(Label(
            text="Image Format:",
            size_hint_y=None,
            height=dp(30)
        ))
        
        format_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(40))
        
        # Format buttons (simple toggle)
        format_buttons = []
        formats = ["png", "jpg"]
        current_format = stats.get('image_format', 'png')
        
        for fmt in formats:
            btn = Button(text=fmt.upper(), size_hint_x=0.5)
            if fmt == current_format:
                btn.background_color = (0, 1, 0, 1)  # Green for selected
            format_buttons.append((btn, fmt))
            format_layout.add_widget(btn)
        
        content.add_widget(format_layout)
        
        # Buttons
        button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(50))
        
        def on_apply(*args):
            try:
                # Get selected format
                selected_format = "png"
                for btn, fmt in format_buttons:
                    if btn.background_color == [0, 1, 0, 1]:
                        selected_format = fmt
                        break
                
                # Apply configuration
                self.intent_predictor.configure_image_saving(image_format=selected_format)
                
                popup.dismiss()
                
                # Show success message
                success_popup = UserMessagePopup(title="Configuration Applied")
                success_popup.ids.message_label.text = f"Frame format set to {selected_format.upper()}"
                success_popup.open()
                
            except Exception as e:
                error_popup = UserMessagePopup(title="Configuration Error")
                error_popup.ids.message_label.text = f"Error: {e}"
                error_popup.open()
        
        def on_cancel(*args):
            popup.dismiss()
        
        def toggle_format(btn, fmt):
            def _toggle(*args):
                # Reset all buttons
                for b, _ in format_buttons:
                    b.background_color = (1, 1, 1, 1)  # Default
                # Set selected button
                btn.background_color = (0, 1, 0, 1)  # Green
            return _toggle
        
        # Bind format button events
        for btn, fmt in format_buttons:
            btn.bind(on_press=toggle_format(btn, fmt))
        
        apply_btn = Button(text="Apply")
        apply_btn.bind(on_press=on_apply)
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_press=on_cancel)
        
        button_layout.add_widget(cancel_btn)
        button_layout.add_widget(apply_btn)
        content.add_widget(button_layout)
        
        popup = Popup(
            title="Frame Saving Configuration",
            content=content,
            size_hint=(0.8, 0.7)
        )
        popup.open()


if __name__ == "__main__":
    app = G3App()
    asyncio.run(app.async_run())