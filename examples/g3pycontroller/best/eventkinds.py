from enum import Enum


class AppEventKind(Enum):
    START_DISCOVERY = "start_discovery"
    ENTER_CONTROL_SESSION = "enter_control_session"
    LEAVE_CONTROL_SESSION = "leave_control_session"
    STOP = "stop"


class ControlEventKind(Enum):
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    DELETE_RECORDING = "delete_recording"
    PLAY_RECORDING = "play_recording"
    START_LIVE = "start_live"
    STOP_LIVE = "stop_live"
    START_HEATMAP = "start_heatmap"
    STOP_HEATMAP = "stop_heatmap"
    START_INTENT_PREDICTION = "start_intent_prediction"
    STOP_INTENT_PREDICTION = "stop_intent_prediction"
    CONFIGURE_INTENT_PREDICTION = "configure_intent_prediction"