"""
OpenCV Video Processing Module
Handles video feeds from CCTV/traffic cameras
"""
import cv2
import numpy as np
from typing import Generator, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from threading import Thread, Lock
from queue import Queue
import time

import sys
sys.path.append('..')
from config import VIDEO_FPS, FRAME_SKIP, RESIZE_WIDTH, RESIZE_HEIGHT


@dataclass
class VideoFrame:
    """Represents a video frame with metadata"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    source: str


class VideoProcessor:
    """
    OpenCV-based video processor for traffic camera feeds
    Supports RTSP streams, video files, and webcams
    """
    
    def __init__(self, source: str = None, 
                 frame_skip: int = FRAME_SKIP,
                 resize: Tuple[int, int] = (RESIZE_WIDTH, RESIZE_HEIGHT)):
        """
        Initialize video processor
        
        Args:
            source: Video source (RTSP URL, file path, or camera index)
            frame_skip: Process every N frames
            resize: Target frame size (width, height)
        """
        self.source = source
        self.frame_skip = frame_skip
        self.resize = resize
        self.cap = None
        self.frame_count = 0
        self.fps = VIDEO_FPS
        self._lock = Lock()
        self._running = False
    
    def connect(self, source: str = None) -> bool:
        """
        Connect to video source
        
        Args:
            source: Video source (overrides init source)
        
        Returns:
            True if connection successful
        """
        source = source or self.source
        if source is None:
            print("No video source specified")
            return False
        
        # Handle different source types
        if isinstance(source, int) or source.isdigit():
            # Webcam
            self.cap = cv2.VideoCapture(int(source))
        elif source.startswith('rtsp://') or source.startswith('http://'):
            # RTSP/HTTP stream
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            # File path
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            print(f"Failed to connect to: {source}")
            return False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
        self.source = source
        print(f"✓ Connected to: {source} ({self.fps:.1f} FPS)")
        return True
    
    def read_frame(self) -> Optional[VideoFrame]:
        """
        Read and process a single frame
        
        Returns:
            VideoFrame or None if no frame available
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        with self._lock:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            self.frame_count += 1
            
            # Resize if specified
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            
            return VideoFrame(
                frame=frame,
                frame_number=self.frame_count,
                timestamp=time.time(),
                source=str(self.source)
            )
    
    def get_frames(self, max_frames: int = None) -> Generator[VideoFrame, None, None]:
        """
        Generator that yields frames from the video source
        
        Args:
            max_frames: Maximum frames to yield (None for infinite)
        
        Yields:
            VideoFrame objects
        """
        self._running = True
        count = 0
        
        while self._running:
            if max_frames and count >= max_frames:
                break
            
            video_frame = self.read_frame()
            if video_frame is None:
                break
            
            # Skip frames for efficiency
            if video_frame.frame_number % self.frame_skip == 0:
                yield video_frame
                count += 1
        
        self._running = False
    
    def stop(self):
        """Stop frame generation"""
        self._running = False
    
    def release(self):
        """Release video capture resources"""
        self._running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()


class AsyncVideoStream:
    """
    Asynchronous video stream reader using threading
    Maintains a buffer of recent frames for processing
    """
    
    def __init__(self, source: str, buffer_size: int = 10):
        """
        Initialize async video stream
        
        Args:
            source: Video source
            buffer_size: Frame buffer size
        """
        self.source = source
        self.buffer = Queue(maxsize=buffer_size)
        self.processor = VideoProcessor(source)
        self._thread = None
        self._running = False
    
    def start(self) -> 'AsyncVideoStream':
        """Start the async reader thread"""
        if not self.processor.connect():
            raise RuntimeError(f"Cannot connect to {self.source}")
        
        self._running = True
        self._thread = Thread(target=self._read_frames, daemon=True)
        self._thread.start()
        return self
    
    def _read_frames(self):
        """Background thread for reading frames"""
        while self._running:
            frame = self.processor.read_frame()
            if frame is None:
                break
            
            # Skip if buffer is full (drop old frames)
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                except:
                    pass
            
            self.buffer.put(frame)
    
    def read(self) -> Optional[VideoFrame]:
        """
        Read the latest frame from buffer
        
        Returns:
            VideoFrame or None
        """
        if self.buffer.empty():
            return None
        return self.buffer.get()
    
    def stop(self):
        """Stop the async reader"""
        self._running = False
        self.processor.release()
        if self._thread:
            self._thread.join(timeout=2)


class MultiCameraManager:
    """
    Manages multiple camera feeds simultaneously
    """
    
    def __init__(self):
        self.cameras: Dict[str, AsyncVideoStream] = {}
        self._lock = Lock()
    
    def add_camera(self, camera_id: str, source: str) -> bool:
        """
        Add a camera feed
        
        Args:
            camera_id: Unique identifier for the camera
            source: Video source URL/path
        
        Returns:
            True if camera added successfully
        """
        with self._lock:
            if camera_id in self.cameras:
                print(f"Camera {camera_id} already exists")
                return False
            
            try:
                stream = AsyncVideoStream(source)
                stream.start()
                self.cameras[camera_id] = stream
                print(f"✓ Added camera: {camera_id}")
                return True
            except Exception as e:
                print(f"✗ Failed to add camera {camera_id}: {e}")
                return False
    
    def remove_camera(self, camera_id: str):
        """Remove a camera feed"""
        with self._lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()
                del self.cameras[camera_id]
    
    def get_frame(self, camera_id: str) -> Optional[VideoFrame]:
        """Get latest frame from a specific camera"""
        with self._lock:
            if camera_id in self.cameras:
                return self.cameras[camera_id].read()
        return None
    
    def get_all_frames(self) -> Dict[str, VideoFrame]:
        """Get latest frames from all cameras"""
        frames = {}
        with self._lock:
            for camera_id, stream in self.cameras.items():
                frame = stream.read()
                if frame:
                    frames[camera_id] = frame
        return frames
    
    def stop_all(self):
        """Stop all camera feeds"""
        with self._lock:
            for stream in self.cameras.values():
                stream.stop()
            self.cameras.clear()


def create_test_frame(width: int = RESIZE_WIDTH, 
                      height: int = RESIZE_HEIGHT,
                      add_vehicles: bool = True) -> np.ndarray:
    """
    Create a synthetic test frame for testing
    
    Args:
        width: Frame width
        height: Frame height
        add_vehicles: Add simulated vehicle shapes
    
    Returns:
        Synthetic frame (BGR)
    """
    # Road background
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (50, 50, 50)  # Dark gray asphalt
    
    # Draw road lines
    cv2.line(frame, (width//4, 0), (width//4, height), (255, 255, 255), 2)
    cv2.line(frame, (width*3//4, 0), (width*3//4, height), (255, 255, 255), 2)
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 255), 2, cv2.LINE_AA)
    
    if add_vehicles:
        import random
        # Add random vehicle rectangles
        for _ in range(random.randint(3, 10)):
            x = random.randint(50, width - 100)
            y = random.randint(50, height - 80)
            w = random.randint(40, 80)
            h = random.randint(30, 50)
            color = random.choice([
                (200, 200, 200),  # Silver
                (0, 0, 150),      # Red
                (150, 0, 0),      # Blue
                (0, 100, 0),      # Green
                (50, 50, 50)      # Black
            ])
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 1)
    
    return frame


if __name__ == "__main__":
    # Test video processing
    print("Testing VideoProcessor...")
    
    # Create and save test frame
    test_frame = create_test_frame()
    cv2.imwrite("test_frame.jpg", test_frame)
    print(f"Test frame created: {test_frame.shape}")
    
    # Test with webcam (if available)
    processor = VideoProcessor(source="0", frame_skip=5)
    if processor.connect():
        print("Webcam connected, reading 10 frames...")
        for i, frame in enumerate(processor.get_frames(max_frames=10)):
            print(f"  Frame {frame.frame_number}: {frame.frame.shape}")
        processor.release()
    else:
        print("No webcam available")
