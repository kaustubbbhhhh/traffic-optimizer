"""Models package for Traffic Flow Optimizer"""
from .vehicle_detection import VehicleDetector, DetectedVehicle, DetectionResult, EmergencyVehicleDetector
from .traffic_forecasting import TrafficForecaster, ForecastResult, generate_sample_data
from .incident_detection import IncidentDetector, Incident, LaneDetector, LaneInfo, EmergencyVehicleTracker
from .signal_coordination import (
    MultiIntersectionCoordinator, 
    Intersection, 
    GreenWave, 
    SignalPhase,
    AdaptiveSignalController,
    create_demo_network
)

__all__ = [
    'VehicleDetector',
    'DetectedVehicle', 
    'DetectionResult',
    'EmergencyVehicleDetector',
    'TrafficForecaster',
    'ForecastResult',
    'generate_sample_data',
    'IncidentDetector',
    'Incident',
    'LaneDetector',
    'LaneInfo',
    'EmergencyVehicleTracker',
    'MultiIntersectionCoordinator',
    'Intersection',
    'GreenWave',
    'SignalPhase',
    'AdaptiveSignalController',
    'create_demo_network'
]
