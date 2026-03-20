"""
Multi-Intersection Coordination Module
Synchronizes nearby traffic signals for green wave optimization
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time


class SignalPhase(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    EMERGENCY = "emergency"


@dataclass
class Intersection:
    """Represents a traffic intersection"""
    id: str
    name: str
    position: Tuple[float, float]  # lat, lon or x, y
    current_phase: SignalPhase = SignalPhase.RED
    phase_start_time: float = field(default_factory=time.time)
    green_duration: int = 30
    yellow_duration: int = 3
    red_duration: int = 30
    vehicle_count: int = 0
    density_level: str = "low"
    emergency_override: bool = False
    
    def get_remaining_time(self) -> int:
        """Get remaining time in current phase"""
        elapsed = time.time() - self.phase_start_time
        if self.current_phase == SignalPhase.GREEN:
            return max(0, int(self.green_duration - elapsed))
        elif self.current_phase == SignalPhase.YELLOW:
            return max(0, int(self.yellow_duration - elapsed))
        else:
            return max(0, int(self.red_duration - elapsed))


@dataclass  
class GreenWave:
    """Represents a coordinated green wave across intersections"""
    id: str
    direction: str  # 'north', 'south', 'east', 'west'
    intersections: List[str]  # ordered list of intersection IDs
    offset_times: Dict[str, int]  # offset in seconds for each intersection
    active: bool = True
    speed_kmh: float = 50.0  # target travel speed
    created_at: datetime = field(default_factory=datetime.now)


class MultiIntersectionCoordinator:
    """
    Coordinates multiple intersections for optimal traffic flow
    Implements green wave synchronization
    """
    
    def __init__(self):
        self.intersections: Dict[str, Intersection] = {}
        self.green_waves: Dict[str, GreenWave] = {}
        self.coordination_groups: Dict[str, List[str]] = {}
    
    def add_intersection(self, intersection: Intersection):
        """Add an intersection to the system"""
        self.intersections[intersection.id] = intersection
    
    def create_intersection(self, id: str, name: str, 
                           position: Tuple[float, float]) -> Intersection:
        """Create and add a new intersection"""
        intersection = Intersection(id=id, name=name, position=position)
        self.add_intersection(intersection)
        return intersection
    
    def update_intersection_density(self, intersection_id: str, 
                                    vehicle_count: int, 
                                    density_level: str):
        """Update traffic density for an intersection"""
        if intersection_id in self.intersections:
            self.intersections[intersection_id].vehicle_count = vehicle_count
            self.intersections[intersection_id].density_level = density_level
            self._optimize_timing(intersection_id)
    
    def _optimize_timing(self, intersection_id: str):
        """Dynamically optimize signal timing based on density"""
        intersection = self.intersections.get(intersection_id)
        if not intersection:
            return
        
        # Adjust timing based on density
        timing_config = {
            'low': {'green': 20, 'red': 40},
            'medium': {'green': 30, 'red': 30},
            'high': {'green': 45, 'red': 20},
            'critical': {'green': 60, 'red': 15}
        }
        
        config = timing_config.get(intersection.density_level, timing_config['medium'])
        intersection.green_duration = config['green']
        intersection.red_duration = config['red']
    
    def create_green_wave(self, wave_id: str, direction: str,
                         intersection_ids: List[str],
                         speed_kmh: float = 50.0) -> GreenWave:
        """
        Create a coordinated green wave across intersections
        
        Args:
            wave_id: Unique identifier for the wave
            direction: Direction of traffic flow
            intersection_ids: Ordered list of intersection IDs
            speed_kmh: Target travel speed in km/h
        
        Returns:
            Created GreenWave object
        """
        # Calculate offset times based on distance and speed
        offset_times = {}
        cumulative_time = 0
        
        for i, int_id in enumerate(intersection_ids):
            offset_times[int_id] = cumulative_time
            
            # Calculate time to next intersection
            if i < len(intersection_ids) - 1:
                current = self.intersections.get(int_id)
                next_int = self.intersections.get(intersection_ids[i + 1])
                
                if current and next_int:
                    distance = self._calculate_distance(
                        current.position, next_int.position
                    )
                    # Convert speed to m/s and calculate travel time
                    speed_ms = speed_kmh * 1000 / 3600
                    travel_time = distance / speed_ms if speed_ms > 0 else 10
                    cumulative_time += int(travel_time)
                else:
                    cumulative_time += 15  # Default 15 seconds between intersections
        
        wave = GreenWave(
            id=wave_id,
            direction=direction,
            intersections=intersection_ids,
            offset_times=offset_times,
            speed_kmh=speed_kmh
        )
        
        self.green_waves[wave_id] = wave
        return wave
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calculate distance between two positions in meters"""
        # Simple Euclidean distance (for demo)
        # In production, would use haversine formula for lat/lon
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx*dx + dy*dy) * 100  # Scale factor
    
    def synchronize_signals(self, wave_id: str, current_time: float = None):
        """
        Synchronize all signals in a green wave
        
        Args:
            wave_id: ID of the green wave to synchronize
            current_time: Current time (defaults to now)
        """
        wave = self.green_waves.get(wave_id)
        if not wave or not wave.active:
            return
        
        current_time = current_time or time.time()
        
        for int_id in wave.intersections:
            intersection = self.intersections.get(int_id)
            if not intersection or intersection.emergency_override:
                continue
            
            offset = wave.offset_times.get(int_id, 0)
            
            # Calculate which phase should be active
            cycle_length = (intersection.green_duration + 
                          intersection.yellow_duration + 
                          intersection.red_duration)
            
            adjusted_time = (current_time + offset) % cycle_length
            
            if adjusted_time < intersection.green_duration:
                intersection.current_phase = SignalPhase.GREEN
            elif adjusted_time < intersection.green_duration + intersection.yellow_duration:
                intersection.current_phase = SignalPhase.YELLOW
            else:
                intersection.current_phase = SignalPhase.RED
    
    def activate_emergency_corridor(self, intersection_ids: List[str]):
        """
        Activate emergency corridor - all specified intersections go green
        
        Args:
            intersection_ids: List of intersections to clear
        """
        for int_id in intersection_ids:
            intersection = self.intersections.get(int_id)
            if intersection:
                intersection.emergency_override = True
                intersection.current_phase = SignalPhase.EMERGENCY
    
    def deactivate_emergency_corridor(self, intersection_ids: List[str]):
        """Deactivate emergency corridor and resume normal operation"""
        for int_id in intersection_ids:
            intersection = self.intersections.get(int_id)
            if intersection:
                intersection.emergency_override = False
                intersection.current_phase = SignalPhase.RED
    
    def get_coordination_status(self) -> Dict:
        """Get current status of all coordinated intersections"""
        status = {
            'intersections': {},
            'green_waves': {},
            'active_emergencies': []
        }
        
        for int_id, intersection in self.intersections.items():
            status['intersections'][int_id] = {
                'name': intersection.name,
                'phase': intersection.current_phase.value,
                'remaining_time': intersection.get_remaining_time(),
                'vehicle_count': intersection.vehicle_count,
                'density': intersection.density_level,
                'emergency': intersection.emergency_override
            }
            
            if intersection.emergency_override:
                status['active_emergencies'].append(int_id)
        
        for wave_id, wave in self.green_waves.items():
            status['green_waves'][wave_id] = {
                'direction': wave.direction,
                'intersections': wave.intersections,
                'active': wave.active,
                'speed_kmh': wave.speed_kmh
            }
        
        return status
    
    def get_signal_recommendations(self) -> List[Dict]:
        """
        Get signal timing recommendations based on current conditions
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for imbalanced traffic
        densities = [(int_id, i.density_level) 
                    for int_id, i in self.intersections.items()]
        
        critical_count = sum(1 for _, d in densities if d == 'critical')
        
        if critical_count > len(densities) * 0.3:
            recommendations.append({
                'type': 'network_congestion',
                'severity': 'high',
                'message': f'{critical_count} intersections at critical density',
                'action': 'Consider activating all green waves'
            })
        
        # Check for green wave opportunities
        for int_id, intersection in self.intersections.items():
            if intersection.density_level in ['high', 'critical']:
                # Find nearby intersections
                nearby = self._find_nearby_intersections(int_id, 500)  # 500m radius
                
                if len(nearby) >= 2:
                    nearby_in_wave = any(
                        int_id in wave.intersections 
                        for wave in self.green_waves.values()
                    )
                    
                    if not nearby_in_wave:
                        recommendations.append({
                            'type': 'green_wave_opportunity',
                            'severity': 'medium',
                            'message': f'Consider green wave for {intersection.name}',
                            'action': f'Coordinate with {nearby}'
                        })
        
        return recommendations
    
    def _find_nearby_intersections(self, int_id: str, 
                                   radius_meters: float) -> List[str]:
        """Find intersections within a given radius"""
        center = self.intersections.get(int_id)
        if not center:
            return []
        
        nearby = []
        for other_id, other in self.intersections.items():
            if other_id != int_id:
                distance = self._calculate_distance(center.position, other.position)
                if distance <= radius_meters:
                    nearby.append(other_id)
        
        return nearby


class AdaptiveSignalController:
    """
    Adaptive signal timing based on real-time traffic conditions
    """
    
    def __init__(self, coordinator: MultiIntersectionCoordinator):
        self.coordinator = coordinator
        self.history = {}
        self.learning_rate = 0.1
    
    def calculate_optimal_timing(self, intersection_id: str,
                                 current_vehicles: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate optimal signal timing based on vehicle counts per direction
        
        Args:
            intersection_id: Target intersection
            current_vehicles: Dict of direction -> vehicle count
        
        Returns:
            Optimal timing configuration
        """
        total_vehicles = sum(current_vehicles.values())
        
        if total_vehicles == 0:
            return {'green': 20, 'yellow': 3, 'red': 40, 'cycle': 63}
        
        # Calculate proportional green times
        timings = {}
        min_green = 10
        max_green = 60
        min_cycle = 60
        max_cycle = 120
        
        # Base cycle time on total volume
        if total_vehicles < 20:
            cycle_time = min_cycle
        elif total_vehicles > 60:
            cycle_time = max_cycle
        else:
            cycle_time = min_cycle + (total_vehicles - 20) * (max_cycle - min_cycle) / 40
        
        # Allocate green time proportionally
        for direction, count in current_vehicles.items():
            proportion = count / total_vehicles if total_vehicles > 0 else 0.25
            green_time = int(proportion * (cycle_time - 12))  # Reserve 12s for yellow phases
            green_time = max(min_green, min(max_green, green_time))
            timings[direction] = {
                'green': green_time,
                'yellow': 3
            }
        
        return {
            'phases': timings,
            'cycle': int(cycle_time),
            'total_vehicles': total_vehicles
        }
    
    def apply_adaptive_timing(self, intersection_id: str, timing: Dict):
        """Apply calculated timing to intersection"""
        intersection = self.coordinator.intersections.get(intersection_id)
        if intersection and not intersection.emergency_override:
            # Apply average green time
            phases = timing.get('phases', {})
            if phases:
                avg_green = np.mean([p['green'] for p in phases.values()])
                intersection.green_duration = int(avg_green)


def create_demo_network() -> MultiIntersectionCoordinator:
    """Create a demo intersection network for testing"""
    coordinator = MultiIntersectionCoordinator()
    
    # Create a grid of intersections
    intersections = [
        ('INT_001', 'Main St & 1st Ave', (0, 0)),
        ('INT_002', 'Main St & 2nd Ave', (100, 0)),
        ('INT_003', 'Main St & 3rd Ave', (200, 0)),
        ('INT_004', 'Main St & 4th Ave', (300, 0)),
        ('INT_005', 'Oak St & 1st Ave', (0, 100)),
        ('INT_006', 'Oak St & 2nd Ave', (100, 100)),
    ]
    
    for int_id, name, pos in intersections:
        coordinator.create_intersection(int_id, name, pos)
    
    # Create a green wave along Main St
    coordinator.create_green_wave(
        'wave_main_east',
        'east',
        ['INT_001', 'INT_002', 'INT_003', 'INT_004'],
        speed_kmh=50
    )
    
    return coordinator


if __name__ == "__main__":
    # Test the coordinator
    coordinator = create_demo_network()
    
    # Update some densities
    coordinator.update_intersection_density('INT_001', 25, 'high')
    coordinator.update_intersection_density('INT_002', 15, 'medium')
    coordinator.update_intersection_density('INT_003', 35, 'critical')
    
    # Synchronize green wave
    coordinator.synchronize_signals('wave_main_east')
    
    # Get status
    status = coordinator.get_coordination_status()
    print("\nIntersection Status:")
    for int_id, data in status['intersections'].items():
        print(f"  {int_id}: {data['phase']} ({data['remaining_time']}s remaining) - {data['density']}")
    
    # Get recommendations
    recommendations = coordinator.get_signal_recommendations()
    print("\nRecommendations:")
    for rec in recommendations:
        print(f"  [{rec['severity']}] {rec['message']}")
