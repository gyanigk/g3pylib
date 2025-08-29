import asyncio
import base64
import json
import logging
import time
import threading
from datetime import datetime
from pathlib import Path
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
        
        # Image saving configuration
        self.save_images_enabled = False
        self.save_directory = Path("saved_frames")
        self.save_scene_frames = True
        self.save_heatmap_frames = True
        self.save_combined_frames = True  # Scene + heatmap side by side
        self.image_format = "png"  # png or jpg
        self.saved_frame_counter = 0
        self._save_lock = threading.Lock()
        
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
    
    def enable_image_saving(self, save_directory: str = "saved_frames"):
        """Enable saving of frames to disk."""
        self.save_images_enabled = True
        self.save_directory = Path(save_directory)
        
        # Create main directory only (no subdirectories)
        self.save_directory.mkdir(exist_ok=True)
        
        # Initialize metadata list
        self._all_predictions_metadata = []
        
        logging.info(f"Image saving enabled - directory: {self.save_directory} (scene + heatmap frames only)")
    
    def disable_image_saving(self):
        """Disable saving of frames to disk."""
        self.save_images_enabled = False
        logging.info("Image saving disabled")
    
    def configure_image_saving(self, image_format: str = "png"):
        """Configure image format for prediction frames."""
        self.image_format = image_format.lower() if image_format.lower() in ["png", "jpg", "jpeg"] else "png"
        
        # Always save scene and heatmap frames for predictions
        self.save_scene_frames = True
        self.save_heatmap_frames = True
        self.save_combined_frames = False  # No longer saving combined frames
        
        logging.info(f"Image saving configured - format: {self.image_format} (saves scene and heatmap frames for predictions)")
    
    def update_frames(self, scene_frame: np.ndarray, heatmap_frame: np.ndarray):
        """Update the latest scene and heatmap frames (thread-safe)."""
        with self._frame_lock:
            self.latest_scene_frame = scene_frame.copy() if scene_frame is not None else None
            self.latest_heatmap_frame = heatmap_frame.copy() if heatmap_frame is not None else None
        
        # Note: Frames are only saved when actually sent to OpenAI for prediction
        # See _save_prediction_frames method in predict_intent methods
    

    
    def _prepare_frame_for_saving(self, frame: np.ndarray) -> np.ndarray:
        """Prepare frame for saving (convert BGR to RGB if needed)."""
        try:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB for proper color saving
                if frame.dtype == np.uint8:
                    # Manual BGR to RGB conversion
                    frame_rgb = frame[:, :, ::-1]  # Reverse the color channel order
                    return frame_rgb
                else:
                    return frame.astype(np.uint8)
            else:
                return frame.astype(np.uint8)
        except Exception as e:
            logging.error(f"Failed to prepare frame for saving: {e}")
            return frame
    
    def _save_single_frame(self, frame: np.ndarray, filename: Path):
        """Save a single frame to disk."""
        try:
            pil_image = Image.fromarray(frame)
            pil_image.save(filename, format=self.image_format.upper() if self.image_format == "png" else "JPEG")
        except Exception as e:
            logging.error(f"Failed to save frame to {filename}: {e}")
    
    def _create_combined_frame(self, scene_frame: np.ndarray, heatmap_frame: np.ndarray) -> np.ndarray:
        """Create a combined frame with scene and heatmap side by side."""
        try:
            # Ensure both frames have same height
            h1, w1 = scene_frame.shape[:2]
            h2, w2 = heatmap_frame.shape[:2]
            
            target_height = min(h1, h2)
            
            # Resize frames to same height if needed
            if h1 != target_height:
                from PIL import Image as PILImage
                scene_pil = PILImage.fromarray(scene_frame)
                scene_pil = scene_pil.resize((int(w1 * target_height / h1), target_height))
                scene_frame = np.array(scene_pil)
            if h2 != target_height:
                from PIL import Image as PILImage
                heatmap_pil = PILImage.fromarray(heatmap_frame)
                heatmap_pil = heatmap_pil.resize((int(w2 * target_height / h2), target_height))
                heatmap_frame = np.array(heatmap_pil)
            
            # Concatenate horizontally
            combined = np.hstack((scene_frame, heatmap_frame))
            return combined
            
        except Exception as e:
            logging.error(f"Failed to create combined frame: {e}")
            return scene_frame  # Return scene frame as fallback
    
    def _save_prediction_frames(self, scene_frame: np.ndarray, heatmap_frame: np.ndarray, timestamp: float):
        """Save frames that are about to be sent to OpenAI for prediction."""
        try:
            with self._save_lock:
                # Increment frame counter for this prediction
                self.saved_frame_counter += 1
                
                # Generate prediction-specific filename
                dt = datetime.fromtimestamp(timestamp)
                timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                prediction_id = f"prediction_{timestamp_str}_{self.saved_frame_counter:06d}"
                
                # Save directly in the main directory (no subdirectory)
                save_dir = self.save_directory
                save_dir.mkdir(exist_ok=True)
                
                # Prepare frames for saving
                scene_frame_save = self._prepare_frame_for_saving(scene_frame)
                heatmap_frame_save = self._prepare_frame_for_saving(heatmap_frame)
                
                # Save scene frame only
                scene_filename = save_dir / f"scene_{prediction_id}.{self.image_format}"
                self._save_single_frame(scene_frame_save, scene_filename)
                
                # Save heatmap frame only
                heatmap_filename = save_dir / f"heatmap_{prediction_id}.{self.image_format}"
                self._save_single_frame(heatmap_frame_save, heatmap_filename)
                
                # Create metadata entry for this prediction (to be saved later)
                prediction_metadata = {
                    "prediction_id": prediction_id,
                    "frame_counter": self.saved_frame_counter,
                    "timestamp": timestamp,
                    "formatted_time": dt.isoformat(),
                    "scene_filename": f"scene_{prediction_id}.{self.image_format}",
                    "heatmap_filename": f"heatmap_{prediction_id}.{self.image_format}",
                    "scene_shape": scene_frame.shape,
                    "heatmap_shape": heatmap_frame.shape,
                    "candidates": self.current_candidates.copy(),
                    "prediction_interval": self.prediction_interval
                }
                
                # Store this prediction metadata to be saved to single file later
                if not hasattr(self, '_all_predictions_metadata'):
                    self._all_predictions_metadata = []
                self._all_predictions_metadata.append(prediction_metadata)
                
                logging.info(f"Saved prediction frames: {prediction_id}")
                
        except Exception as e:
            logging.error(f"Failed to save prediction frames: {e}")
    
    def _save_prediction_result(self, prediction: Dict[str, Any]):
        """Save the prediction result to the metadata and write the complete metadata file."""
        try:
            if not self.save_images_enabled or not hasattr(self, '_all_predictions_metadata'):
                return
                
            with self._save_lock:
                # Find the most recent metadata entry and add prediction result
                if self._all_predictions_metadata:
                    latest_metadata = self._all_predictions_metadata[-1]
                    latest_metadata.update({
                        "prediction": prediction.get("prediction", "unknown"),
                        "reasoning": prediction.get("reasoning", ""),
                        "duration_ms": prediction.get("duration_ms", 0),
                        "prediction_timestamp": prediction.get("timestamp", 0),
                        "prediction_start_time": prediction.get("start_time", 0),
                        "prediction_end_time": prediction.get("end_time", 0)
                    })
                    
                    # Save complete metadata file
                    metadata_file = self.save_directory / "predictions_metadata.json"
                    complete_metadata = {
                        "session_info": {
                            "total_predictions": len(self._all_predictions_metadata),
                            "created_at": datetime.now().isoformat(),
                            "prediction_interval": self.prediction_interval,
                            "image_format": self.image_format
                        },
                        "predictions": self._all_predictions_metadata
                    }
                    
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(complete_metadata, f, indent=2, default=str)
                    
                    logging.debug(f"Updated metadata with prediction result: {prediction.get('prediction', 'unknown')}")
                
        except Exception as e:
            logging.error(f"Failed to save prediction result to metadata: {e}")
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert numpy frame to base64 encoded PNG."""
        try:
            # Ensure frame is in correct format (RGB)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if frame.dtype == np.uint8:
                    # Manual BGR to RGB conversion to avoid cv2 linter issues
                    frame_rgb = frame[:, :, ::-1]  # Reverse the color channel order
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
            
            # Save frames to disk if image saving is enabled (before sending to API)
            if self.save_images_enabled:
                try:
                    self._save_prediction_frames(scene_frame, heatmap_frame, current_time)
                except Exception as e:
                    logging.error(f"Failed to save prediction frames: {e}")
            
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
            
            # Save prediction result to metadata if image saving is enabled
            self._save_prediction_result(prediction)
            
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
            
            # Save frames to disk if image saving is enabled (before sending to API)
            if self.save_images_enabled:
                try:
                    self._save_prediction_frames(scene_frame, heatmap_frame, timestamp)
                except Exception as e:
                    logging.error(f"Failed to save prediction frames: {e}")
            
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
            
            # Save prediction result to metadata if image saving is enabled
            self._save_prediction_result(prediction)
            
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
    
    def get_image_saving_stats(self) -> Dict[str, Any]:
        """Get image saving statistics."""
        return {
            "save_images_enabled": self.save_images_enabled,
            "save_directory": str(self.save_directory),
            "save_scene_frames": self.save_scene_frames,
            "save_heatmap_frames": self.save_heatmap_frames,
            "save_combined_frames": self.save_combined_frames,
            "image_format": self.image_format,
            "saved_frame_counter": self.saved_frame_counter,
            "total_predictions": len(getattr(self, '_all_predictions_metadata', [])),
        }
    
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
            "image_saving": self.get_image_saving_stats(),
        }
