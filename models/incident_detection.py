"""
AI Incident Detection Module
Detects accidents, stalled vehicles, and abnormal traffic events
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class Incident:
    """Represents a detected incident"""
    incident_type: str  # 'accident', 'stalled_vehicle', 'congestion_spike', 'wrong_way', 'pedestrian_on_road'
    confidence: float
    location: Tuple[int, int]  # x, y coordinates
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str


@dataclass
class LaneInfo:
    """Information about a traffic lane"""
    lane_id: int
    vehicle_count: int
    density: str
    average_speed: float  # estimated
    vehicles: List[dict]


class IncidentDetector:
    """
    AI-powered incident detection system
    Detects accidents, stalled vehicles, congestion spikes, etc.
    """
    
    def __init__(self):
        self.vehicle_history = deque(maxlen=30)  # 30 frames history
        self.position_history = {}  # Track vehicle positions over time
        self.congestion_baseline = None
        self.last_incident_time = {}
        self.incident_cooldown = 10  # seconds between same incident type
    
    def detect_incidents(self, frame: np.ndarray, 
                         vehicles: List[dict],
                         current_density: str) -> List[Incident]:
        """
        Detect various traffic incidents
        
        Args:
            frame: Current video frame
            vehicles: List of detected vehicles with bboxes
            current_density: Current traffic density level
        
        Returns:
            List of detected incidents
        """
        incidents = []
        current_time = datetime.now()
        
        # Store current state
        self.vehicle_history.append({
            'timestamp': current_time,
            'vehicles': vehicles,
            'density': current_density,
            'count': len(vehicles)
        })
        
        # 1. Detect stalled vehicles (vehicles not moving)
        stalled = self._detect_stalled_vehicles(vehicles)
        incidents.extend(stalled)
        
        # 2. Detect sudden congestion spikes
        congestion = self._detect_congestion_spike()
        if congestion:
            incidents.append(congestion)
        
        # 3. Detect potential accidents (multiple stalled vehicles close together)
        accidents = self._detect_accidents(vehicles, frame)
        incidents.extend(accidents)
        
        # 4. Detect abnormal vehicle behavior
        abnormal = self._detect_abnormal_behavior(vehicles)
        incidents.extend(abnormal)
        
        return incidents
    
    def _detect_stalled_vehicles(self, vehicles: List[dict]) -> List[Incident]:
        """Detect vehicles that haven't moved for extended period"""
        incidents = []
        
        for vehicle in vehicles:
            vehicle_id = self._get_vehicle_id(vehicle)
            center = vehicle.get('center', (0, 0))
            
            if vehicle_id in self.position_history:
                history = self.position_history[vehicle_id]
                history.append({'pos': center, 'time': time.time()})
                
                # Keep last 30 positions
                if len(history) > 30:
                    history.pop(0)
                
                # Check if vehicle hasn't moved significantly
                if len(history) >= 20:
                    positions = [h['pos'] for h in history]
                    movement = self._calculate_movement(positions)
                    
                    if movement < 10:  # Less than 10 pixels movement
                        time_span = history[-1]['time'] - history[0]['time']
                        if time_span > 5:  # Stalled for more than 5 seconds
                            incidents.append(Incident(
                                incident_type='stalled_vehicle',
                                confidence=0.85,
                                location=center,
                                bbox=vehicle.get('bbox', (0, 0, 0, 0)),
                                timestamp=datetime.now(),
                                severity='medium',
                                description=f"Vehicle stalled at position {center}"
                            ))
            else:
                self.position_history[vehicle_id] = [{'pos': center, 'time': time.time()}]
        
        return incidents
    
    def _detect_congestion_spike(self) -> Optional[Incident]:
        """Detect sudden increase in vehicle count"""
        if len(self.vehicle_history) < 10:
            return None
        
        recent = list(self.vehicle_history)
        counts = [h['count'] for h in recent]
        
        # Calculate average of first half vs second half
        mid = len(counts) // 2
        first_avg = np.mean(counts[:mid])
        second_avg = np.mean(counts[mid:])
        
        # Spike if count increased by more than 50%
        if first_avg > 0 and second_avg > first_avg * 1.5:
            if self._can_report_incident('congestion_spike'):
                return Incident(
                    incident_type='congestion_spike',
                    confidence=0.75,
                    location=(320, 240),
                    bbox=(0, 0, 640, 480),
                    timestamp=datetime.now(),
                    severity='high',
                    description=f"Sudden congestion spike: {first_avg:.0f} → {second_avg:.0f} vehicles"
                )
        
        return None
    
    def _detect_accidents(self, vehicles: List[dict], frame: np.ndarray) -> List[Incident]:
        """Detect potential accidents based on vehicle clustering and positions"""
        incidents = []
        
        if len(vehicles) < 2:
            return incidents
        
        # Find clusters of stalled vehicles
        positions = [v.get('center', (0, 0)) for v in vehicles]
        
        for i, v1 in enumerate(vehicles):
            for j, v2 in enumerate(vehicles[i+1:], i+1):
                pos1 = v1.get('center', (0, 0))
                pos2 = v2.get('center', (0, 0))
                
                # Check if vehicles are very close (potential collision)
                distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if distance < 30:  # Very close vehicles
                    # Check for irregular angles (not aligned with traffic flow)
                    if self._can_report_incident('accident'):
                        center = ((pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2)
                        incidents.append(Incident(
                            incident_type='accident',
                            confidence=0.6,
                            location=center,
                            bbox=(min(pos1[0], pos2[0]) - 50, min(pos1[1], pos2[1]) - 50,
                                  max(pos1[0], pos2[0]) + 50, max(pos1[1], pos2[1]) + 50),
                            timestamp=datetime.now(),
                            severity='critical',
                            description="Potential collision detected - vehicles in close proximity"
                        ))
        
        return incidents
    
    def _detect_abnormal_behavior(self, vehicles: List[dict]) -> List[Incident]:
        """Detect abnormal vehicle behavior patterns"""
        incidents = []
        
        # This would use trajectory analysis in a full implementation
        # For now, detect vehicles in unusual positions
        
        for vehicle in vehicles:
            bbox = vehicle.get('bbox', (0, 0, 0, 0))
            x1, y1, x2, y2 = bbox
            
            # Check if vehicle is at edge of frame (potential wrong way)
            # This is a simplified check
            
        return incidents
    
    def _get_vehicle_id(self, vehicle: dict) -> str:
        """Generate a pseudo-ID for vehicle tracking"""
        center = vehicle.get('center', (0, 0))
        cls = vehicle.get('class_name', 'unknown')
        # Simple hash based on position (for demo - real system would use proper tracking)
        return f"{cls}_{center[0]//50}_{center[1]//50}"
    
    def _calculate_movement(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate total movement distance"""
        if len(positions) < 2:
            return 0
        
        total = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)
        
        return total
    
    def _can_report_incident(self, incident_type: str) -> bool:
        """Check if cooldown period has passed for incident type"""
        current_time = time.time()
        last_time = self.last_incident_time.get(incident_type, 0)
        
        if current_time - last_time > self.incident_cooldown:
            self.last_incident_time[incident_type] = current_time
            return True
        return False


class LaneDetector:
    """
    Detects and analyzes traffic lanes
    """
    
    def __init__(self, num_lanes: int = 4):
        self.num_lanes = num_lanes
        self.lane_boundaries = []
    
    def setup_lanes(self, frame_width: int, frame_height: int):
        """Setup lane boundaries based on frame dimensions"""
        lane_width = frame_width // self.num_lanes
        self.lane_boundaries = []
        
        for i in range(self.num_lanes):
            self.lane_boundaries.append({
                'id': i,
                'x_start': i * lane_width,
                'x_end': (i + 1) * lane_width,
                'y_start': 0,
                'y_end': frame_height
            })
    
    def analyze_lanes(self, vehicles: List[dict], 
                      frame_width: int = 640,
                      frame_height: int = 480) -> List[LaneInfo]:
        """
        Analyze traffic in each lane
        
        Args:
            vehicles: List of detected vehicles
            frame_width: Frame width
            frame_height: Frame height
        
        Returns:
            List of LaneInfo for each lane
        """
        if not self.lane_boundaries:
            self.setup_lanes(frame_width, frame_height)
        
        lane_data = []
        
        for lane in self.lane_boundaries:
            lane_vehicles = []
            
            for v in vehicles:
                center = v.get('center', (0, 0))
                if lane['x_start'] <= center[0] < lane['x_end']:
                    lane_vehicles.append(v)
            
            count = len(lane_vehicles)
            
            # Determine density based on count
            if count < 2:
                density = 'low'
            elif count < 5:
                density = 'medium'
            elif count < 8:
                density = 'high'
            else:
                density = 'critical'
            
            lane_data.append(LaneInfo(
                lane_id=lane['id'],
                vehicle_count=count,
                density=density,
                average_speed=self._estimate_speed(lane_vehicles),
                vehicles=lane_vehicles
            ))
        
        return lane_data
    
    def _estimate_speed(self, vehicles: List[dict]) -> float:
        """Estimate average speed based on vehicle spacing"""
        if len(vehicles) < 2:
            return 50.0  # Default speed
        
        # More vehicles = slower speed (simplified model)
        return max(10, 60 - len(vehicles) * 5)
    
    def draw_lanes(self, frame: np.ndarray, lane_info: List[LaneInfo]) -> np.ndarray:
        """Draw lane overlays on frame"""
        overlay = frame.copy()
        
        colors = {
            'low': (0, 255, 0),      # Green
            'medium': (0, 255, 255),  # Yellow
            'high': (0, 165, 255),    # Orange
            'critical': (0, 0, 255)   # Red
        }
        
        for lane in lane_info:
            if lane.lane_id < len(self.lane_boundaries):
                bounds = self.lane_boundaries[lane.lane_id]
                color = colors.get(lane.density, (128, 128, 128))
                
                # Draw semi-transparent overlay
                cv2.rectangle(overlay, 
                             (bounds['x_start'], 0),
                             (bounds['x_end'], frame.shape[0]),
                             color, -1)
                
                # Add lane label
                label = f"L{lane.lane_id + 1}: {lane.vehicle_count}"
                cv2.putText(overlay, label, 
                           (bounds['x_start'] + 5, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original
        return cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)


class EmergencyVehicleTracker:
    """
    Tracks emergency vehicles and calculates optimal corridor
    """
    
    def __init__(self):
        self.active_emergencies = {}
        self.corridor_signals = []
    
    def track_emergency(self, vehicle_id: str, position: Tuple[int, int],
                        vehicle_type: str, direction: str):
        """Track an emergency vehicle"""
        self.active_emergencies[vehicle_id] = {
            'position': position,
            'type': vehicle_type,
            'direction': direction,
            'timestamp': datetime.now(),
            'corridor_active': True
        }
    
    def calculate_corridor(self, emergency_position: Tuple[int, int],
                          intersections: List[dict]) -> List[str]:
        """
        Calculate which signals need to be cleared for emergency corridor
        
        Args:
            emergency_position: Current position of emergency vehicle
            intersections: List of intersection data
        
        Returns:
            List of intersection IDs to clear
        """
        # In a real system, this would use GPS routing
        # For demo, return next 3 intersections
        clear_signals = []
        
        for intersection in intersections[:3]:
            clear_signals.append(intersection.get('id', 'INT_001'))
        
        self.corridor_signals = clear_signals
        return clear_signals
    
    def get_active_corridors(self) -> List[dict]:
        """Get all active emergency corridors"""
        return [
            {
                'vehicle_id': vid,
                'data': data,
                'signals': self.corridor_signals
            }
            for vid, data in self.active_emergencies.items()
            if data.get('corridor_active', False)
        ]


if __name__ == "__main__":
    # Test incident detector
    detector = IncidentDetector()
    
    # Simulate some vehicles
    vehicles = [
        {'center': (100, 200), 'bbox': (80, 180, 120, 220), 'class_name': 'car'},
        {'center': (105, 205), 'bbox': (85, 185, 125, 225), 'class_name': 'car'},
        {'center': (300, 200), 'bbox': (280, 180, 320, 220), 'class_name': 'truck'},
    ]
    
    incidents = detector.detect_incidents(np.zeros((480, 640, 3)), vehicles, 'medium')
    print(f"Detected {len(incidents)} incidents")
    for inc in incidents:
        print(f"  - {inc.incident_type}: {inc.description}")
