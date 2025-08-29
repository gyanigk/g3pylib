import asyncio
import base64
import json
import logging
import time
import threading
# from pathlib import Path  # Unused for now
from typing import List, Dict, Optional, Any
import io
from collections import deque

import cv2
import numpy as np
from PIL import Image

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI library not available. Install with: pip install openai")


class IntentPredictor:
    """Real-time intent prediction using gaze data and scene frames."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.client = None
        
        # Configuration
        self.prediction_interval = 2.0  # Seconds between predictions
        self.max_completion_tokens = 500
        self.temperature = 0.3
        
        # Default candidate actions for coffee/kitchen scene
        self.default_candidates = [
            "grab pink cup", "grab white cup", "grab orange juice", "grab steel cup",
            "grab sugar packet", "grab yellow packet", "grab half half milk", "grab silk milk",
            "grab soy milk", "grab soda can", "pour half half milk", "pour silk milk",
            "pour soy milk", "pour soda can", "pour steel cup", "pour sugar packet",
            "place cup", "place half half milk", "place silk milk", "place soy milk",
            "place steel cup", "place orange juice", "place pink cup", "place soda can", 
            "observing", "nothing"
        ]
        
        # State tracking
        self.is_enabled = False
        self.last_prediction_time = 0
        self.prediction_history = deque(maxlen=10)
        self.current_candidates = self.default_candidates.copy()
        
        # Frame buffers (thread-safe access)
        self.latest_scene_frame = None
        self.latest_heatmap_frame = None
        self._frame_lock = threading.Lock()
        
        # Latest prediction storage for UI updates
        self._latest_prediction = None
        self._prediction_lock = threading.Lock()
        
        # System prompt for intent prediction
        self.system_prompt = """
            You are an expert AI assistant specializing in human intent recognition. Your primary mission is to analyze a visual scene and its corresponding eye-gaze data to predict the most likely action a person will take next.

            Your Inputs:
            - Scene Image: An RGB image showing the environment and objects.
            - Gaze Visualization Image: A heatmap overlay showing where the user looked the longest or most frequently (bright/hot areas indicate focus areas).

            Your Task & Rules:
            - Carefully analyze both the scene and the gaze visualization to understand what the user is focusing on. The gaze data is your primary clue to their intent.
            - From the provided candidates list, select the single most probable action.
            - If the gaze data is broad, scattered across multiple objects without settling, or appears to be a general survey of the scene, it indicates the user is still observing or planning. In this case, select "observing".
            - Your final output must be a JSON object containing two keys:
            - "prediction": The string of the action you selected from the list.
            - "reasoning": A brief, one-sentence explanation of why you made your choice, specifically referencing the gaze data.
            - Always return your response as a valid JSON string, e.g., {"prediction": "action", "reasoning": "explanation"}.
        """
        
        # Initialize OpenAI client if available
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                logging.info("IntentPredictor initialized with OpenAI client")
            except Exception as e:
                logging.error(f"Failed to initialize OpenAI client: {e}")
        elif not OPENAI_AVAILABLE:
            logging.warning("OpenAI not available - intent prediction disabled")
        else:
            logging.warning("No API key provided - intent prediction disabled")
    
    def set_api_key(self, api_key: str):
        """Set or update the OpenAI API key."""
        self.api_key = api_key
        if OPENAI_AVAILABLE:
            try:
                self.client = OpenAI(api_key=api_key)
                logging.info("OpenAI client updated with new API key")
            except Exception as e:
                logging.error(f"Failed to update OpenAI client: {e}")
    
    def enable_prediction(self):
        """Enable real-time intent prediction."""
        if not self.client:
            logging.warning("Cannot enable prediction - no OpenAI client available")
            return False
        self.is_enabled = True
        logging.info("Intent prediction enabled")
        return True
    
    def disable_prediction(self):
        """Disable real-time intent prediction."""
        self.is_enabled = False
        logging.info("Intent prediction disabled")
    
    def set_candidates(self, candidates: List[str]):
        """Set the list of candidate actions."""
        if not candidates:
            logging.warning("Empty candidates list - using defaults")
            self.current_candidates = self.default_candidates.copy()
        else:
            self.current_candidates = candidates.copy()
            # Ensure 'observing' is always available
            if "observing" not in self.current_candidates:
                self.current_candidates.append("observing")
        logging.info(f"Updated candidates: {len(self.current_candidates)} actions")
    
    def set_prediction_interval(self, interval: float):
        """Set the interval between predictions in seconds."""
        self.prediction_interval = max(0.5, min(interval, 10.0))  # Limit between 0.5 and 10 seconds
        logging.info(f"Prediction interval set to {self.prediction_interval} seconds")
    
    def update_frames(self, scene_frame: np.ndarray, heatmap_frame: np.ndarray):
        """Update the latest scene and heatmap frames (thread-safe)."""
        with self._frame_lock:
            self.latest_scene_frame = scene_frame.copy() if scene_frame is not None else None
            self.latest_heatmap_frame = heatmap_frame.copy() if heatmap_frame is not None else None
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 encoded PNG."""
        try:
            # Ensure frame is in correct format (RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if frame.dtype == np.uint8:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
            else:
                raise ValueError(f"Invalid frame shape: {frame.shape}")
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb.astype(np.uint8))
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to convert frame to base64: {e}")
            raise
    
    def _make_prediction_request(self, scene_b64: str, heatmap_b64: str) -> Dict[str, Any]:
        """Make a prediction request to the OpenAI API."""
        user_text = f"""
            Analyze the provided scene and gaze data to predict the user's intended action.
            Gaze Representation Type: heatmap
            [Image 1: Scene RGB]
            [Image 2: Gaze Heatmap Visualization]
            Candidate Actions:
            {{
            "actions": {json.dumps(self.current_candidates)}
            }}
            Provide your prediction in the required JSON format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{scene_b64}"}},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{heatmap_b64}"}},
                        ],
                    },
                ],
            )
            
            response_content = response.choices[0].message.content or "{}"
            logging.debug(f"Raw API response: {response_content}")
            
            # Clean response to remove Markdown code block delimiters
            cleaned_response = response_content.strip()
            if cleaned_response.startswith("```json") and cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith("```") and cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[3:-3].strip()
            
            # Parse JSON response
            result = json.loads(cleaned_response)
            
            if not isinstance(result, dict) or "prediction" not in result or "reasoning" not in result:
                raise ValueError(f"Invalid response format: {result}")
            
            return result
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parse error: {e}, response: {cleaned_response}")
            return {"prediction": "observing", "reasoning": f"Failed to parse response: {str(e)}"}
        except Exception as e:
            logging.error(f"Prediction request failed: {e}")
            return {"prediction": "observing", "reasoning": f"API error: {str(e)}"}
    
    def predict_intent(self) -> Optional[Dict[str, Any]]:
        """Predict user intent based on current frames (synchronous version)."""
        current_time = time.time()
        
        # Check if prediction is enabled and enough time has passed
        if not self.is_enabled:
            return None
        
        if current_time - self.last_prediction_time < self.prediction_interval:
            return None
        
        # Check if we have valid frames
        with self._frame_lock:
            if self.latest_scene_frame is None or self.latest_heatmap_frame is None:
                logging.debug("No frames available for prediction")
                return None
            
            # Copy frames to avoid holding lock during prediction
            scene_frame = self.latest_scene_frame.copy()
            heatmap_frame = self.latest_heatmap_frame.copy()
        
        if not self.client:
            logging.warning("No OpenAI client available for prediction")
            return None
        
        try:
            start_time = time.time()
            
            # Convert frames to base64
            scene_b64 = self._frame_to_base64(scene_frame)
            heatmap_b64 = self._frame_to_base64(heatmap_frame)
            
            # Make prediction
            prediction = self._make_prediction_request(scene_b64, heatmap_b64)
            
            # Calculate timing
            end_time = time.time()
            prediction_duration = end_time - start_time
            
            # Add timestamp, timing and store in history
            prediction["timestamp"] = current_time
            prediction["duration_ms"] = round(prediction_duration * 1000, 2)
            prediction["start_time"] = start_time
            prediction["end_time"] = end_time
            self.prediction_history.append(prediction)
            self.last_prediction_time = current_time
            
            # Store latest prediction for UI updates
            with self._prediction_lock:
                self._latest_prediction = prediction
            
            logging.info(f"Intent prediction: {prediction['prediction']} - {prediction['reasoning']} (took {prediction_duration*1000:.2f}ms)")
            return prediction
            
        except Exception as e:
            logging.error(f"Failed to predict intent: {e}")
            error_prediction = {
                "prediction": "observing", 
                "reasoning": f"Prediction failed: {str(e)}", 
                "timestamp": current_time,
                "duration_ms": 0,
                "start_time": current_time,
                "end_time": current_time
            }
            with self._prediction_lock:
                self._latest_prediction = error_prediction
            return error_prediction
    
    async def predict_intent_async(self) -> Optional[Dict[str, Any]]:
        """Predict user intent based on current frames (async version)."""
        current_time = time.time()
        
        # Check if prediction is enabled and enough time has passed
        if not self.is_enabled:
            return None
        
        if current_time - self.last_prediction_time < self.prediction_interval:
            return None
        
        # Check if we have valid frames
        with self._frame_lock:
            if self.latest_scene_frame is None or self.latest_heatmap_frame is None:
                logging.debug("No frames available for prediction")
                return None
            
            # Copy frames to avoid holding lock during prediction
            scene_frame = self.latest_scene_frame.copy()
            heatmap_frame = self.latest_heatmap_frame.copy()
        
        if not self.client:
            logging.warning("No OpenAI client available for prediction")
            return None
        
        try:
            # Run prediction in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None, self._predict_intent_sync, scene_frame, heatmap_frame, current_time
            )
            
            # Store latest prediction for UI updates
            with self._prediction_lock:
                self._latest_prediction = prediction
            
            logging.info(f"Intent prediction: {prediction['prediction']} - {prediction['reasoning']}")
            return prediction
            
        except Exception as e:
            logging.error(f"Failed to predict intent: {e}")
            error_prediction = {"prediction": "observing", "reasoning": f"Prediction failed: {str(e)}", "timestamp": current_time}
            with self._prediction_lock:
                self._latest_prediction = error_prediction
            return error_prediction
    
    def _predict_intent_sync(self, scene_frame: np.ndarray, heatmap_frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Synchronous prediction helper for executor."""
        try:
            start_time = time.time()
            
            # Convert frames to base64
            scene_b64 = self._frame_to_base64(scene_frame)
            heatmap_b64 = self._frame_to_base64(heatmap_frame)
            
            # Make prediction
            prediction = self._make_prediction_request(scene_b64, heatmap_b64)
            
            # Calculate timing
            end_time = time.time()
            prediction_duration = end_time - start_time
            
            # Add timestamp, timing and store in history
            prediction["timestamp"] = timestamp
            prediction["duration_ms"] = round(prediction_duration * 1000, 2)
            prediction["start_time"] = start_time
            prediction["end_time"] = end_time
            self.prediction_history.append(prediction)
            self.last_prediction_time = timestamp
            
            return prediction
            
        except Exception as e:
            logging.error(f"Failed to predict intent in sync helper: {e}")
            return {
                "prediction": "observing", 
                "reasoning": f"Prediction failed: {str(e)}", 
                "timestamp": timestamp,
                "duration_ms": 0,
                "start_time": timestamp,
                "end_time": timestamp
            }
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction."""
        if self.prediction_history:
            return self.prediction_history[-1]
        return None
    
    def get_latest_prediction_safe(self) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction (thread-safe)."""
        with self._prediction_lock:
            return self._latest_prediction
    
    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Get the full prediction history."""
        return list(self.prediction_history)
    
    def clear_history(self):
        """Clear the prediction history."""
        self.prediction_history.clear()
        logging.info("Prediction history cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        return {
            "is_enabled": self.is_enabled,
            "has_client": self.client is not None,
            "prediction_interval": self.prediction_interval,
            "history_length": len(self.prediction_history),
            "candidates_count": len(self.current_candidates),
            "has_frames": self.latest_scene_frame is not None and self.latest_heatmap_frame is not None,
            "last_prediction_time": self.last_prediction_time,
        }
