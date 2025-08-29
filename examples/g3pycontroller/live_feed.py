"""
Glasses3 Live Gaze Feed - Simplified Version

This script provides a live video feed from Tobii Glasses 3 with real-time gaze overlay.
Simplified for better reliability and easier debugging.

Usage:
    python live_feed.py [options]

Options:
    --hostname HOSTNAME       Glasses3 hostname (default: from G3_HOSTNAME env var)
    --verbose, -v            Enable verbose logging
    --skip-calibration       Skip calibration step
    --max-frames N           Maximum frames to process (0 = unlimited)

Examples:
    python live_feed.py                           # Use default hostname
    python live_feed.py --hostname my-glasses     # Specify hostname
    python live_feed.py --verbose                 # Enable debug logging
    python live_feed.py --skip-calibration        # Skip calibration
    python live_feed.py --max-frames 1000         # Process max 1000 frames

Exit:
    Press 'q' in the video window to quit the live feed.
    Or use Ctrl+C in terminal for graceful shutdown.
"""

import asyncio
import logging
import os
import sys
import argparse
from typing import Any, List, cast
import numpy as np
import cv2
import dotenv
from g3pylib import connect_to_glasses
import time

# Constants
GAZE_CIRCLE_RADIUS = 10
LIVE_FRAME_RATE = 25
MAX_QUEUE_SIZE = 100

def setup_logging(verbose: bool = False):
    """Set up logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

async def calibrate_g3(g3):
    """Calibrate Tobii Glasses 3 using a printed marker."""
    try:
        logging.info("Starting calibration...")
        
        # Start rudimentary streams to activate gaze tracking
        await g3.rudimentary.start_streams()
        await asyncio.sleep(0.5)
        
        # Subscribe to marker stream and emit marker
        marker_queue, unsubscribe_to_marker = await g3.calibrate.subscribe_to_marker()
        await g3.calibrate.emit_markers()
        logging.info("Emitting calibration marker. Ensure the printed marker is visible to the scene camera.")
        
        # Wait for marker data
        try:
            marker = cast(List[Any], await asyncio.wait_for(marker_queue.get(), timeout=10.0))
            logging.info(f"Marker detected: {marker}")
        except asyncio.TimeoutError:
            logging.error("Timeout waiting for marker data. Ensure the marker is in view.")
            return False
        finally:
            await unsubscribe_to_marker
        
        logging.info("Calibration completed successfully")
        return True
    
    except Exception as e:
        logging.error(f"Error during calibration: {e}")
        return False

async def live_gaze_feed(g3, max_frames: int = 0):
    """Display live scene camera feed with gaze overlay."""
    try:
        logging.info("Starting RTSP streams...")
        start_time = time.time()
        
        async with g3.stream_rtsp(scene_camera=True, gaze=True) as streams:
            async with streams.scene_camera.decode() as scene_stream, streams.gaze.decode() as gaze_stream:
                logging.info("Started live stream with gaze tracking")
                frame_count = 0
                
                # Test OpenCV display
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(test_frame, "Testing OpenCV Display", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(test_frame, "Press any key to continue...", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Glasses3 Live Gaze Feed", test_frame)
                
                # Wait for key press or timeout
                key = cv2.waitKey(2000)  # Wait 2 seconds
                if key & 0xFF == ord("q"):
                    return
                
                logging.info("Starting main frame processing loop...")
                
                while True:
                    try:
                        # Check if we've reached max frames
                        if max_frames > 0 and frame_count >= max_frames:
                            logging.info(f"Reached maximum frame limit ({max_frames}), stopping...")
                            break
                        
                        # Get latest frame and gaze data
                        frame_data = await scene_stream.get()
                        gaze_data = await gaze_stream.get()
                        
                        if frame_data is None or gaze_data is None:
                            logging.warning(f"Frame {frame_count}: Null data received")
                            continue
                        
                        # Extract frame and gaze
                        frame, frame_timestamp = frame_data
                        gaze, gaze_timestamp = gaze_data
                        
                        # Process frame
                        frame_array = frame.to_ndarray(format="bgr24")
                        if frame_array is None or frame_array.size == 0:
                            logging.warning(f"Invalid frame data at frame {frame_count}")
                            continue
                        
                        if not frame_array.flags['C_CONTIGUOUS']:
                            frame_array = np.ascontiguousarray(frame_array, dtype=np.uint8)
                        
                        height, width = frame_array.shape[:2]
                        
                        # Log frame info occasionally
                        if frame_count % 30 == 0:
                            logging.info(f"Frame {frame_count}: {width}x{height}")
                        
                        # Draw gaze visualization
                        if "gaze2d" in gaze and gaze["gaze2d"]:
                            gaze_point = gaze["gaze2d"]
                            if len(gaze_point) == 2 and all(isinstance(x, (int, float)) for x in gaze_point):
                                x = int(gaze_point[0] * width)
                                y = int(gaze_point[1] * height)
                                
                                # Ensure coordinates are within bounds
                                x = max(0, min(x, width - 1))
                                y = max(0, min(y, height - 1))
                                
                                # Draw gaze circle
                                cv2.circle(frame_array, (x, y), GAZE_CIRCLE_RADIUS, (0, 0, 255), -1)
                                cv2.circle(frame_array, (x, y), GAZE_CIRCLE_RADIUS + 2, (255, 255, 255), 2)
                                
                                if frame_count % 30 == 0:
                                    logging.info(f"Gaze2d: {gaze_point[0]:.4f}, {gaze_point[1]:.4f}")
                        
                        # Add status display
                        cv2.putText(frame_array, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame_array, "Gaze: Active" if "gaze2d" in gaze and gaze["gaze2d"] else "Gaze: No Data", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if "gaze2d" in gaze else (0, 0, 255), 2)
                        
                        # Add timestamp info if available
                        if frame_timestamp is not None:
                            cv2.putText(frame_array, f"Frame TS: {frame_timestamp:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        if gaze_timestamp is not None:
                            cv2.putText(frame_array, f"Gaze TS: {gaze_timestamp:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        # Add max frames info if set
                        if max_frames > 0:
                            cv2.putText(frame_array, f"Max: {max_frames}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Display the frame
                        cv2.imshow("Glasses3 Live Gaze Feed", frame_array)
                        
                        # Check for quit command
                        key = cv2.waitKey(int(1000 / LIVE_FRAME_RATE))
                        if key & 0xFF == ord("q"):
                            logging.info("Quitting live feed")
                            break
                        
                        frame_count += 1
                        
                        # Log performance occasionally
                        if frame_count % 100 == 0:
                            elapsed_time = time.time() - start_time
                            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                            logging.info(f"Performance: {fps:.1f} FPS, {frame_count} frames in {elapsed_time:.1f}s")
                        
                        # Small delay to prevent overwhelming the system
                        await asyncio.sleep(0.01)
                        
                    except Exception as frame_error:
                        logging.error(f"Error processing frame {frame_count}: {frame_error}")
                        continue
                
                logging.info(f"Live feed completed. Total frames processed: {frame_count}")
                            
    except Exception as e:
        logging.error(f"Error in live gaze feed: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
    finally:
        cv2.destroyAllWindows()
        logging.info("OpenCV windows cleaned up")

async def main():
    """Main function to run calibration followed by live gaze feed."""
    parser = argparse.ArgumentParser(description="Glasses3 Live Gaze Feed with Calibration")
    parser.add_argument("--hostname", default=os.environ.get("G3_HOSTNAME", "tg03b-080200009391"),
                       help="Glasses3 hostname (default: from G3_HOSTNAME env var)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--skip-calibration", action="store_true", help="Skip calibration step")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = unlimited)")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    try:
        # Connect to Glasses3 device
        logging.info(f"Attempting to connect to Glasses3 at {args.hostname}...")
        async with connect_to_glasses.with_hostname(args.hostname, using_zeroconf=True) as g3:
            logging.info(f"Connected to Glasses3 at {args.hostname}")
            
            # Run calibration or start streams directly
            if args.skip_calibration:
                logging.info("Skipping calibration, starting rudimentary streams for gaze tracking...")
                await g3.rudimentary.start_streams()
                await asyncio.sleep(0.5)
                await live_gaze_feed(g3, args.max_frames)
            elif await calibrate_g3(g3):
                logging.info("Calibration completed successfully. Starting live gaze feed...")
                await live_gaze_feed(g3, args.max_frames)
            else:
                logging.error("Calibration failed. Live feed will not start.")
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Error in main process: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
    finally:
        logging.info("Shutting down gracefully...")

if __name__ == "__main__":
    dotenv.load_dotenv()
    asyncio.run(main())