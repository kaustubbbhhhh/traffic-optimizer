"""
YOLOv8 Vehicle Detection & Classification Module
Uses Ultralytics YOLOv8 for real-time vehicle detection from traffic camera feeds
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics not installed. Using mock detection.")

import sys
sys.path.append('..')
from config import YOLO_MODEL, YOLO_CONFIDENCE, YOLO_IOU_THRESHOLD, VEHICLE_CLASSES


@dataclass
class DetectedVehicle:
    """Represents a detected vehicle"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int


@dataclass
class DetectionResult:
    """Result of vehicle detection on a frame"""
    frame: np.ndarray
    vehicles: List[DetectedVehicle]
    vehicle_counts: Dict[str, int]
    total_count: int
    processing_time: float


class VehicleDetector:
    """
    YOLOv8-based vehicle detection and classification
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to YOLOv8 model weights (default: yolov8n.pt)
        """
        self.model_path = model_path or YOLO_MODEL
        self.model = None
        self.vehicle_classes = VEHICLE_CLASSES
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the YOLOv8 model"""
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
                print(f"✓ YOLOv8 model loaded: {self.model_path}")
            except Exception as e:
                print(f"✗ Failed to load YOLOv8: {e}")
                self.model = None
        else:
            print("⚠ Running in mock detection mode")
    
    def detect(self, frame: np.ndarray, 
               confidence: float = None,
               draw_boxes: bool = True) -> DetectionResult:
        """
        Detect vehicles in a frame
        
        Args:
            frame: Input image/frame (BGR format)
            confidence: Minimum confidence threshold
            draw_boxes: Whether to draw bounding boxes on the frame
        
        Returns:
            DetectionResult with detected vehicles and annotated frame
        """
        import time
        start_time = time.time()
        
        confidence = confidence or YOLO_CONFIDENCE
        vehicles = []
        vehicle_counts = defaultdict(int)
        annotated_frame = frame.copy()
        
        if self.model is not None:
            # Run YOLOv8 inference
            results = self.model(frame, conf=confidence, iou=YOLO_IOU_THRESHOLD, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    
                    # Only process vehicle classes
                    if class_id in self.vehicle_classes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        class_name = self.vehicle_classes[class_id]
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        area = (x2 - x1) * (y2 - y1)
                        
                        vehicle = DetectedVehicle(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=conf,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                            area=area
                        )
                        vehicles.append(vehicle)
                        vehicle_counts[class_name] += 1
                        
                        if draw_boxes:
                            annotated_frame = self._draw_detection(
                                annotated_frame, vehicle
                            )
        else:
            # Mock detection for testing
            vehicles, vehicle_counts = self._mock_detection(frame)
            if draw_boxes:
                for vehicle in vehicles:
                    annotated_frame = self._draw_detection(annotated_frame, vehicle)
        
        processing_time = time.time() - start_time
        
        return DetectionResult(
            frame=annotated_frame,
            vehicles=vehicles,
            vehicle_counts=dict(vehicle_counts),
            total_count=len(vehicles),
            processing_time=processing_time
        )
    
    def _draw_detection(self, frame: np.ndarray, 
                        vehicle: DetectedVehicle) -> np.ndarray:
        """Draw bounding box and label on frame"""
        x1, y1, x2, y2 = vehicle.bbox
        
        # Color based on vehicle type
        colors = {
            "car": (0, 255, 0),        # Green
            "motorcycle": (255, 165, 0), # Orange
            "bus": (255, 0, 0),         # Blue
            "truck": (0, 0, 255),       # Red
            "bicycle": (255, 255, 0)    # Cyan
        }
        color = colors.get(vehicle.class_name, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{vehicle.class_name}: {vehicle.confidence:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(frame, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def _mock_detection(self, frame: np.ndarray) -> Tuple[List[DetectedVehicle], Dict[str, int]]:
        """Generate mock detections for testing without YOLO"""
        import random
        
        h, w = frame.shape[:2]
        vehicles = []
        counts = defaultdict(int)
        
        # Generate random vehicles
        num_vehicles = random.randint(5, 20)
        for _ in range(num_vehicles):
            class_id = random.choice(list(self.vehicle_classes.keys()))
            class_name = self.vehicle_classes[class_id]
            
            # Random bounding box
            x1 = random.randint(0, w - 100)
            y1 = random.randint(0, h - 100)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(50, 100)
            
            vehicle = DetectedVehicle(
                class_id=class_id,
                class_name=class_name,
                confidence=random.uniform(0.6, 0.99),
                bbox=(x1, y1, min(x2, w), min(y2, h)),
                center=((x1 + x2) // 2, (y1 + y2) // 2),
                area=(x2 - x1) * (y2 - y1)
            )
            vehicles.append(vehicle)
            counts[class_name] += 1
        
        return vehicles, dict(counts)
    
    def get_density_level(self, vehicle_count: int) -> str:
        """
        Determine traffic density level based on vehicle count
        
        Args:
            vehicle_count: Total number of detected vehicles
        
        Returns:
            Density level: 'low', 'medium', 'high', or 'critical'
        """
        from config import DENSITY_THRESHOLDS
        
        if vehicle_count < DENSITY_THRESHOLDS["low"]:
            return "low"
        elif vehicle_count < DENSITY_THRESHOLDS["medium"]:
            return "medium"
        elif vehicle_count < DENSITY_THRESHOLDS["high"]:
            return "high"
        else:
            return "critical"


class EmergencyVehicleDetector:
    """
    Detects emergency vehicles using color analysis
    (ambulances, fire trucks, police vehicles)
    """
    
    def __init__(self):
        from config import EMERGENCY_COLORS
        self.emergency_colors = EMERGENCY_COLORS
    
    def detect(self, frame: np.ndarray, 
               vehicles: List[DetectedVehicle]) -> List[DetectedVehicle]:
        """
        Identify emergency vehicles from detected vehicles
        
        Args:
            frame: Original frame (BGR)
            vehicles: List of detected vehicles
        
        Returns:
            List of vehicles identified as emergency vehicles
        """
        emergency_vehicles = []
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            roi = hsv_frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Check for emergency colors (red/blue lights)
            is_emergency = False
            for color_name, (lower, upper) in self.emergency_colors.items():
                mask = cv2.inRange(roi, np.array(lower), np.array(upper))
                ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])
                
                # If significant portion has emergency colors
                if ratio > 0.1:  # 10% threshold
                    is_emergency = True
                    break
            
            if is_emergency:
                emergency_vehicles.append(vehicle)
        
        return emergency_vehicles


if __name__ == "__main__":
    # Test the detector
    detector = VehicleDetector()
    
    # Create a test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Gray background
    
    result = detector.detect(test_frame)
    print(f"Detected {result.total_count} vehicles")
    print(f"Vehicle counts: {result.vehicle_counts}")
    print(f"Density level: {detector.get_density_level(result.total_count)}")
    print(f"Processing time: {result.processing_time*1000:.2f}ms")
