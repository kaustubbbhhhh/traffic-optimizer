"""
Data Handling & Analytics Module
Uses NumPy and Pandas for traffic data processing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

import sys
sys.path.append('..')
from config import DATA_DIR, DENSITY_THRESHOLDS


@dataclass
class TrafficStats:
    """Aggregated traffic statistics"""
    total_vehicles: int
    avg_vehicles_per_hour: float
    peak_hour: int
    peak_count: int
    density_distribution: Dict[str, float]
    trend: str  # 'increasing', 'decreasing', 'stable'


class TrafficDataHandler:
    """
    Handles traffic data storage, retrieval, and analytics
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize data handler
        
        Args:
            data_dir: Directory for data storage
        """
        self.data_dir = data_dir or str(DATA_DIR)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # In-memory data store (can be replaced with database)
        self.traffic_data = pd.DataFrame(columns=[
            'timestamp', 'intersection_id', 'direction',
            'vehicle_count', 'car', 'motorcycle', 'bus', 'truck', 'bicycle',
            'density_level', 'emergency_detected'
        ])
        
        # Load existing data
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV storage"""
        data_file = os.path.join(self.data_dir, 'traffic_history.csv')
        if os.path.exists(data_file):
            try:
                self.traffic_data = pd.read_csv(data_file, parse_dates=['timestamp'])
                print(f"✓ Loaded {len(self.traffic_data)} traffic records")
            except Exception as e:
                print(f"✗ Failed to load data: {e}")
    
    def save_data(self):
        """Save data to CSV storage"""
        data_file = os.path.join(self.data_dir, 'traffic_history.csv')
        self.traffic_data.to_csv(data_file, index=False)
    
    def add_record(self, intersection_id: str, direction: str,
                   vehicle_counts: Dict[str, int], 
                   emergency_detected: bool = False,
                   timestamp: datetime = None):
        """
        Add a traffic record
        
        Args:
            intersection_id: ID of the intersection
            direction: Traffic direction (N, S, E, W)
            vehicle_counts: Dict with vehicle type counts
            emergency_detected: Emergency vehicle flag
            timestamp: Record timestamp
        """
        timestamp = timestamp or datetime.now()
        total = sum(vehicle_counts.values())
        density = self._get_density_level(total)
        
        record = {
            'timestamp': timestamp,
            'intersection_id': intersection_id,
            'direction': direction,
            'vehicle_count': total,
            'car': vehicle_counts.get('car', 0),
            'motorcycle': vehicle_counts.get('motorcycle', 0),
            'bus': vehicle_counts.get('bus', 0),
            'truck': vehicle_counts.get('truck', 0),
            'bicycle': vehicle_counts.get('bicycle', 0),
            'density_level': density,
            'emergency_detected': emergency_detected
        }
        
        self.traffic_data = pd.concat([
            self.traffic_data, 
            pd.DataFrame([record])
        ], ignore_index=True)
    
    def _get_density_level(self, vehicle_count: int) -> str:
        """Determine density level"""
        if vehicle_count < DENSITY_THRESHOLDS["low"]:
            return "low"
        elif vehicle_count < DENSITY_THRESHOLDS["medium"]:
            return "medium"
        elif vehicle_count < DENSITY_THRESHOLDS["high"]:
            return "high"
        return "critical"
    
    def get_recent_data(self, hours: int = 24, 
                        intersection_id: str = None) -> pd.DataFrame:
        """
        Get recent traffic data
        
        Args:
            hours: Number of hours to look back
            intersection_id: Filter by intersection
        
        Returns:
            Filtered DataFrame
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        df = self.traffic_data[self.traffic_data['timestamp'] >= cutoff]
        
        if intersection_id:
            df = df[df['intersection_id'] == intersection_id]
        
        return df.sort_values('timestamp')
    
    def get_hourly_aggregates(self, date: datetime = None) -> pd.DataFrame:
        """
        Get hourly traffic aggregates
        
        Args:
            date: Specific date (default: today)
        
        Returns:
            Hourly aggregated data
        """
        date = date or datetime.now()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        df = self.traffic_data[
            (self.traffic_data['timestamp'] >= start) & 
            (self.traffic_data['timestamp'] < end)
        ].copy()
        
        if len(df) == 0:
            return pd.DataFrame()
        
        df['hour'] = df['timestamp'].dt.hour
        
        return df.groupby('hour').agg({
            'vehicle_count': 'mean',
            'car': 'mean',
            'motorcycle': 'mean',
            'bus': 'mean',
            'truck': 'mean',
            'bicycle': 'mean',
            'emergency_detected': 'sum'
        }).reset_index()
    
    def get_statistics(self, intersection_id: str = None, 
                       hours: int = 24) -> TrafficStats:
        """
        Calculate traffic statistics
        
        Args:
            intersection_id: Filter by intersection
            hours: Time period to analyze
        
        Returns:
            TrafficStats object
        """
        df = self.get_recent_data(hours, intersection_id)
        
        if len(df) == 0:
            return TrafficStats(
                total_vehicles=0,
                avg_vehicles_per_hour=0,
                peak_hour=0,
                peak_count=0,
                density_distribution={'low': 1.0, 'medium': 0, 'high': 0, 'critical': 0},
                trend='stable'
            )
        
        # Hourly aggregates
        df['hour'] = df['timestamp'].dt.hour
        hourly = df.groupby('hour')['vehicle_count'].mean()
        
        # Density distribution
        density_counts = df['density_level'].value_counts(normalize=True)
        density_dist = {
            level: density_counts.get(level, 0) 
            for level in ['low', 'medium', 'high', 'critical']
        }
        
        # Trend analysis (compare first half vs second half)
        mid = len(df) // 2
        first_half = df.iloc[:mid]['vehicle_count'].mean()
        second_half = df.iloc[mid:]['vehicle_count'].mean()
        
        if second_half > first_half * 1.1:
            trend = 'increasing'
        elif second_half < first_half * 0.9:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return TrafficStats(
            total_vehicles=int(df['vehicle_count'].sum()),
            avg_vehicles_per_hour=df['vehicle_count'].mean(),
            peak_hour=int(hourly.idxmax()) if len(hourly) > 0 else 0,
            peak_count=int(hourly.max()) if len(hourly) > 0 else 0,
            density_distribution=density_dist,
            trend=trend
        )
    
    def get_vehicle_distribution(self, hours: int = 24) -> Dict[str, int]:
        """Get vehicle type distribution"""
        df = self.get_recent_data(hours)
        
        return {
            'car': int(df['car'].sum()),
            'motorcycle': int(df['motorcycle'].sum()),
            'bus': int(df['bus'].sum()),
            'truck': int(df['truck'].sum()),
            'bicycle': int(df['bicycle'].sum())
        }
    
    def get_intersection_comparison(self) -> pd.DataFrame:
        """Compare traffic across intersections"""
        if len(self.traffic_data) == 0:
            return pd.DataFrame()
        
        return self.traffic_data.groupby('intersection_id').agg({
            'vehicle_count': ['mean', 'max', 'sum'],
            'emergency_detected': 'sum'
        }).reset_index()
    
    def export_to_json(self, filepath: str = None) -> str:
        """Export data to JSON format"""
        filepath = filepath or os.path.join(self.data_dir, 'traffic_export.json')
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'record_count': len(self.traffic_data),
            'data': self.traffic_data.to_dict(orient='records')
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, default=str, indent=2)
        
        return filepath


class SignalOptimizer:
    """
    Optimizes traffic signal timing based on current conditions
    """
    
    def __init__(self):
        from config import SIGNAL_TIMING
        self.timing_config = SIGNAL_TIMING
    
    def calculate_timing(self, density_level: str, 
                        emergency_active: bool = False) -> Dict[str, int]:
        """
        Calculate optimal signal timing
        
        Args:
            density_level: Current traffic density
            emergency_active: Emergency vehicle present
        
        Returns:
            Dict with green, yellow, red timings
        """
        if emergency_active:
            # Emergency override - extend green significantly
            return {'green': 90, 'yellow': 3, 'red': 5}
        
        return self.timing_config.get(density_level, self.timing_config['medium'])
    
    def optimize_multi_direction(self, 
                                 direction_densities: Dict[str, str]) -> Dict[str, Dict[str, int]]:
        """
        Optimize signals for multiple directions
        
        Args:
            direction_densities: Dict mapping direction to density level
        
        Returns:
            Optimized timings per direction
        """
        timings = {}
        
        # Prioritize higher density directions
        priority_order = ['critical', 'high', 'medium', 'low']
        
        for direction, density in direction_densities.items():
            base_timing = self.calculate_timing(density)
            
            # Adjust based on relative priority
            density_rank = priority_order.index(density) if density in priority_order else 2
            
            # Higher density gets more green time
            multiplier = 1 + (3 - density_rank) * 0.1
            
            timings[direction] = {
                'green': int(base_timing['green'] * multiplier),
                'yellow': base_timing['yellow'],
                'red': base_timing['red']
            }
        
        return timings


def generate_demo_data(handler: TrafficDataHandler, 
                       days: int = 7,
                       intersections: List[str] = None):
    """
    Generate demo traffic data for testing
    
    Args:
        handler: TrafficDataHandler instance
        days: Number of days of data
        intersections: List of intersection IDs
    """
    intersections = intersections or ['INT_001', 'INT_002', 'INT_003', 'INT_004']
    directions = ['N', 'S', 'E', 'W']
    
    np.random.seed(42)
    
    start_time = datetime.now() - timedelta(days=days)
    current_time = start_time
    
    while current_time < datetime.now():
        for int_id in intersections:
            for direction in directions:
                hour = current_time.hour
                day_of_week = current_time.weekday()
                
                # Realistic traffic pattern
                if day_of_week < 5:  # Weekday
                    if 7 <= hour <= 9 or 17 <= hour <= 19:
                        base = 30
                    elif 10 <= hour <= 16:
                        base = 18
                    elif 22 <= hour or hour <= 5:
                        base = 4
                    else:
                        base = 12
                else:  # Weekend
                    if 10 <= hour <= 18:
                        base = 15
                    else:
                        base = 6
                
                # Add randomness
                base = max(0, int(base + np.random.normal(0, base * 0.3)))
                
                # Vehicle type distribution
                vehicle_counts = {
                    'car': int(base * 0.7),
                    'motorcycle': int(base * 0.1),
                    'bus': int(base * 0.05),
                    'truck': int(base * 0.1),
                    'bicycle': int(base * 0.05)
                }
                
                # Rare emergency events
                emergency = np.random.random() < 0.002
                
                handler.add_record(
                    intersection_id=int_id,
                    direction=direction,
                    vehicle_counts=vehicle_counts,
                    emergency_detected=emergency,
                    timestamp=current_time
                )
        
        current_time += timedelta(hours=1)
    
    handler.save_data()
    print(f"✓ Generated {len(handler.traffic_data)} demo records")


if __name__ == "__main__":
    # Test data handler
    handler = TrafficDataHandler()
    
    # Generate demo data
    print("Generating demo data...")
    generate_demo_data(handler, days=3)
    
    # Test statistics
    stats = handler.get_statistics()
    print(f"\nTraffic Statistics:")
    print(f"  Total vehicles: {stats.total_vehicles}")
    print(f"  Avg per hour: {stats.avg_vehicles_per_hour:.1f}")
    print(f"  Peak hour: {stats.peak_hour}:00 ({stats.peak_count} vehicles)")
    print(f"  Trend: {stats.trend}")
    print(f"  Density distribution: {stats.density_distribution}")
    
    # Test vehicle distribution
    dist = handler.get_vehicle_distribution()
    print(f"\nVehicle Distribution: {dist}")
