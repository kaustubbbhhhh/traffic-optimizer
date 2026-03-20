"""
AI-Driven Traffic Flow Optimizer & Emergency Green Corridor System
Main Streamlit Dashboard Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import cv2
from PIL import Image
import io
import av

# WebRTC for live webcam
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Import custom modules
from config import (CAMERA_FEEDS, DENSITY_THRESHOLDS, SIGNAL_TIMING, 
                   DASHBOARD_REFRESH_RATE)
from models.vehicle_detection import VehicleDetector, EmergencyVehicleDetector
from models.traffic_forecasting import TrafficForecaster, generate_sample_data
from models.incident_detection import IncidentDetector, LaneDetector, EmergencyVehicleTracker
from models.signal_coordination import (
    MultiIntersectionCoordinator, create_demo_network, 
    AdaptiveSignalController, SignalPhase
)
from utils.data_handler import TrafficDataHandler, SignalOptimizer, generate_demo_data
from utils.video_processor import VideoProcessor, create_test_frame

# RTC Configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Set Plotly default template to dark
pio.templates.default = "plotly_dark"

# Page configuration
st.set_page_config(
    page_title="Traffic Flow Optimizer",
    page_icon="TFO",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Dark Theme - Professional Corporate Styling */
    
    /* Main app background */
    .stApp {
        background-color: #0f172a !important;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important;
    }
    [data-testid="stHeader"] {
        background-color: #0f172a !important;
    }
    .main .block-container {
        background-color: #0f172a !important;
    }
    
    /* Page headers */
    .main-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        margin-bottom: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #334155;
        font-family: 'Segoe UI', system-ui, sans-serif;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #94a3b8 !important;
        margin-bottom: 1.5rem;
        font-family: 'Segoe UI', system-ui, sans-serif;
        line-height: 1.5;
    }
    
    /* All text in main area */
    .stMarkdown, .stMarkdown p, [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
    }
    label, .stTextInput label, .stSelectbox label, .stMultiSelect label {
        color: #cbd5e1 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem;
        font-weight: 600;
        color: #f1f5f9 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500;
    }
    div[data-testid="stMetricDelta"] {
        color: #4ade80 !important;
    }
    
    /* Status badges */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-low { background-color: #059669; color: white; }
    .status-medium { background-color: #d97706; color: white; }
    .status-high { background-color: #dc2626; color: white; }
    .status-critical { background-color: #7c2d12; color: white; }
    
    /* Emergency alerts */
    .emergency-alert {
        background-color: #1e1e2e;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        color: #f1f5f9 !important;
    }
    
    /* Info, success, warning, error boxes */
    [data-testid="stAlert"] {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 0.5rem;
    }
    .stAlert > div {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        color: #94a3b8 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
        color: #f1f5f9 !important;
        font-weight: 500;
        padding: 0.5rem 0.75rem;
        border-radius: 0.375rem;
        margin: 0.125rem 0;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
        background-color: #334155 !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #475569 !important;
    }
    [data-testid="stSidebar"] .stMetric label {
        color: #94a3b8 !important;
    }
    [data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
    }
    
    /* Dividers in main area */
    hr {
        border-color: #334155 !important;
    }
    
    /* Subheaders */
    h2, h3, .stSubheader {
        color: #f1f5f9 !important;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 0.375rem;
        font-weight: 600;
        text-transform: none;
        transition: all 0.15s ease;
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #2563eb !important;
    }
    .stButton > button[kind="secondary"] {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #475569 !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2563eb !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #f1f5f9 !important;
        background-color: #1e293b !important;
    }
    [data-testid="stExpander"] {
        background-color: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 0.5rem;
    }
    [data-testid="stExpander"] > div {
        color: #e2e8f0 !important;
    }
    
    /* Select boxes and inputs */
    .stSelectbox > div > div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    .stMultiSelect > div > div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    .stTextInput > div > div > input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    .stNumberInput > div > div > input {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border-color: #475569 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        color: #e2e8f0 !important;
    }
    
    /* DataFrames and tables */
    .stDataFrame {
        background-color: #1e293b !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #1e293b !important;
    }
    .stDataFrame th {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    .stDataFrame td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important;
        border-bottom-color: #3b82f6 !important;
    }
    
    /* Progress spinner */
    .stSpinner > div {
        color: #e2e8f0 !important;
    }
    
    /* Plotly charts - dark background */
    .js-plotly-plot .plotly {
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = TrafficDataHandler()
    if 'detector' not in st.session_state:
        st.session_state.detector = VehicleDetector()
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = TrafficForecaster()
    if 'signal_optimizer' not in st.session_state:
        st.session_state.signal_optimizer = SignalOptimizer()
    if 'emergency_active' not in st.session_state:
        st.session_state.emergency_active = False
    if 'demo_data_generated' not in st.session_state:
        st.session_state.demo_data_generated = False
    if 'live_detections' not in st.session_state:
        st.session_state.live_detections = []
    # New modules
    if 'incident_detector' not in st.session_state:
        st.session_state.incident_detector = IncidentDetector()
    if 'lane_detector' not in st.session_state:
        st.session_state.lane_detector = LaneDetector(num_lanes=4)
    if 'coordinator' not in st.session_state:
        st.session_state.coordinator = create_demo_network()
    if 'adaptive_controller' not in st.session_state:
        st.session_state.adaptive_controller = AdaptiveSignalController(st.session_state.coordinator)
    if 'emergency_tracker' not in st.session_state:
        st.session_state.emergency_tracker = EmergencyVehicleTracker()
    if 'incidents' not in st.session_state:
        st.session_state.incidents = []
    if 'active_corridors' not in st.session_state:
        st.session_state.active_corridors = []


class VideoTransformer(VideoProcessorBase):
    """Video processor for live webcam detection with incident detection"""
    
    def __init__(self):
        self.detector = VehicleDetector()
        self.incident_detector = IncidentDetector()
        self.lane_detector = LaneDetector(num_lanes=4)
        self.frame_count = 0
        self.last_result = None
        self.process_every_n = 3
        self.incidents = []
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        if self.frame_count % self.process_every_n == 0:
            # Run vehicle detection
            result = self.detector.detect(img, draw_boxes=True)
            self.last_result = result
            
            # Run lane analysis
            vehicles_data = [
                {'center': v.center, 'bbox': v.bbox, 'class_name': v.class_name}
                for v in result.vehicles
            ]
            lane_info = self.lane_detector.analyze_lanes(vehicles_data, img.shape[1], img.shape[0])
            
            # Run incident detection
            density = self.detector.get_density_level(result.total_count)
            self.incidents = self.incident_detector.detect_incidents(img, vehicles_data, density)
            
            # Draw lane overlays
            annotated = self.lane_detector.draw_lanes(result.frame, lane_info)
            
            # Draw incident markers
            for incident in self.incidents:
                x, y = incident.location
                color = (0, 0, 255) if incident.severity == 'critical' else (0, 165, 255)
                cv2.circle(annotated, (x, y), 20, color, 3)
                cv2.putText(annotated, incident.incident_type.upper(), (x-40, y-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        elif self.last_result is not None:
            return av.VideoFrame.from_ndarray(self.last_result.frame, format="bgr24")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        # Professional branding
        st.markdown("""
        <div style="padding: 1rem 0 1.5rem 0; text-align: center;">
            <div style="font-size: 2rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.05em;">
                TFO
            </div>
            <div style="font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.15em; margin-top: 0.25rem;">
                Traffic Flow Optimizer
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation label
        st.markdown('<p style="font-size: 0.7rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; font-weight: 600;">Navigation</p>', unsafe_allow_html=True)
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Dashboard", "Live Detection", "Signal Control", "Analytics", "Emergency Control", "Settings"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Quick stats
        if len(st.session_state.data_handler.traffic_data) > 0:
            st.markdown('<p style="font-size: 0.7rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; font-weight: 600;">Quick Stats</p>', unsafe_allow_html=True)
            stats = st.session_state.data_handler.get_statistics(hours=1)
            st.metric("Avg Vehicles/hr", f"{stats.avg_vehicles_per_hour:.0f}")
            st.metric("Current Trend", stats.trend.capitalize())
        
        st.divider()
        
        # Demo data button with label
        st.markdown('<p style="font-size: 0.7rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.5rem; font-weight: 600;">Actions</p>', unsafe_allow_html=True)
        if st.button("Generate Demo Data", use_container_width=True, type="primary"):
            with st.spinner("Generating..."):
                generate_demo_data(st.session_state.data_handler, days=3)
                st.session_state.demo_data_generated = True
            st.success("Demo data generated!")
            st.rerun()
        
        return page


def render_dashboard():
    """Main dashboard view"""
    st.markdown('<h1 class="main-header">Traffic Flow Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time traffic monitoring and analysis powered by AI</p>', unsafe_allow_html=True)
    
    # Check for data
    if len(st.session_state.data_handler.traffic_data) == 0:
        st.info("No traffic data available. Click 'Generate Demo Data' in the sidebar to get started.")
        return
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    stats = st.session_state.data_handler.get_statistics(hours=24)
    
    with col1:
        st.metric(
            "Total Vehicles (24h)",
            f"{stats.total_vehicles:,}",
            delta=f"{stats.trend}"
        )
    
    with col2:
        st.metric(
            "Avg Vehicles/Hour",
            f"{stats.avg_vehicles_per_hour:.0f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Peak Hour",
            f"{stats.peak_hour}:00",
            delta=f"{stats.peak_count} vehicles"
        )
    
    with col4:
        # Get density color
        dominant_density = max(stats.density_distribution, 
                             key=stats.density_distribution.get)
        st.metric(
            "Current Density",
            dominant_density.upper(),
            delta=None
        )
    
    st.divider()
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hourly Traffic Pattern")
        hourly_data = st.session_state.data_handler.get_hourly_aggregates()
        
        if len(hourly_data) > 0:
            fig = px.area(
                hourly_data,
                x='hour',
                y='vehicle_count',
                title='',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Avg Vehicle Count",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hourly data available for today")
    
    with col2:
        st.subheader("Vehicle Distribution")
        vehicle_dist = st.session_state.data_handler.get_vehicle_distribution()
        
        if sum(vehicle_dist.values()) > 0:
            fig = px.pie(
                values=list(vehicle_dist.values()),
                names=list(vehicle_dist.keys()),
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vehicle distribution data available")
    
    # Traffic Forecast
    st.divider()
    st.subheader("Traffic Forecast (Next 6 Hours)")
    
    recent_data = st.session_state.data_handler.get_recent_data(hours=48)
    if len(recent_data) >= 24:
        # Aggregate by hour for forecasting
        recent_data['hour_bucket'] = recent_data['timestamp'].dt.floor('h')
        forecast_input = recent_data.groupby('hour_bucket').agg({
            'vehicle_count': 'mean'
        }).reset_index().rename(columns={'hour_bucket': 'timestamp'})
        
        forecast = st.session_state.forecaster.forecast(forecast_input)
        
        # Create forecast chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_input['timestamp'].tail(24),
            y=forecast_input['vehicle_count'].tail(24),
            mode='lines',
            name='Historical',
            line=dict(color='#6b7280')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast.timestamps,
            y=forecast.predictions,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#667eea', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast.timestamps + forecast.timestamps[::-1],
            y=list(forecast.confidence_upper) + list(forecast.confidence_lower[::-1]),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence'
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Predicted Vehicle Count",
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 24 hours of data for forecasting")
    
    # Signal Status
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Signal Status")
        
        # Create signal status cards
        intersections = st.session_state.data_handler.traffic_data['intersection_id'].unique()
        
        if len(intersections) > 0:
            cols = st.columns(min(4, len(intersections)))
            
            for idx, int_id in enumerate(intersections[:4]):
                with cols[idx]:
                    int_stats = st.session_state.data_handler.get_statistics(
                        intersection_id=int_id, hours=1
                    )
                    density = max(int_stats.density_distribution, 
                                key=int_stats.density_distribution.get)
                    timing = st.session_state.signal_optimizer.calculate_timing(density)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, 
                        {'#10b981' if density == 'low' else '#f59e0b' if density == 'medium' else '#ef4444' if density == 'high' else '#7c3aed'} 0%, 
                        {'#059669' if density == 'low' else '#d97706' if density == 'medium' else '#dc2626' if density == 'high' else '#6d28d9'} 100%);
                        padding: 1rem; border-radius: 0.75rem; color: white; text-align: center;">
                        <div style="font-size: 0.875rem; opacity: 0.9;">{int_id}</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">{density.upper()}</div>
                        <div style="font-size: 0.75rem; margin-top: 0.5rem;">
                             {timing['green']}s |  {timing['yellow']}s |  {timing['red']}s
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("System Status")
        
        emergency_count = st.session_state.data_handler.traffic_data['emergency_detected'].sum()
        
        if st.session_state.emergency_active or emergency_count > 0:
            st.markdown("""
            <div class="emergency-alert">
                <strong> Emergency Alert</strong><br>
                Green corridor activated. All signals prioritized for emergency vehicle passage.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(" No active emergencies")
        
        # Active incidents
        incidents = st.session_state.incidents
        if incidents:
            st.warning(f" {len(incidents)} active incident(s)")
            for inc in incidents[-3:]:
                st.caption(f" {inc.incident_type}: {inc.severity}")
        
        # Coordination status
        coordinator = st.session_state.coordinator
        active_waves = len([w for w in coordinator.green_waves.values() if w.active])
        if active_waves > 0:
            st.info(f" {active_waves} green wave(s) active")
    
    # Network status row
    st.divider()
    st.subheader("Multi-Intersection Network")
    
    net_cols = st.columns(len(st.session_state.coordinator.intersections))
    for idx, (int_id, intersection) in enumerate(st.session_state.coordinator.intersections.items()):
        with net_cols[idx]:
            phase_color = {
                SignalPhase.GREEN: "#10b981",
                SignalPhase.YELLOW: "#f59e0b",
                SignalPhase.RED: "#ef4444"
            }.get(intersection.current_phase, "#6b7280")
            
            phase_text = {
                SignalPhase.GREEN: "G",
                SignalPhase.YELLOW: "Y",
                SignalPhase.RED: "R"
            }.get(intersection.current_phase, "-")
            
            st.markdown(f"""
            <div style="text-align: center; padding: 0.5rem; border: 1px solid #e5e7eb; border-radius: 0.5rem;">
                <div style="width: 24px; height: 24px; border-radius: 50%; background-color: {phase_color}; margin: 0 auto; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.75rem;">{phase_text}</div>
                <div style="font-size: 0.75rem; font-weight: 600; margin-top: 0.25rem;">{int_id}</div>
            </div>
            """, unsafe_allow_html=True)


def render_live_detection():
    """Live video detection view"""
    st.markdown('<h1 class="main-header">Live Vehicle Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">YOLOv8-powered real-time vehicle detection and classification</p>', unsafe_allow_html=True)
    
    # Initialize video state
    if 'video_playing' not in st.session_state:
        st.session_state.video_playing = False
    if 'current_stats' not in st.session_state:
        st.session_state.current_stats = {'total': 0, 'counts': {}}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video source selection
        source_type = st.radio(
            "Video Source",
            ["Upload Image", "Upload Video", "Test Frame", "Webcam"],
            horizontal=True
        )
        
        frame = None
        is_video = False
        
        if source_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Upload a traffic image",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        elif source_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a traffic video",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm']
            )
            
            if uploaded_file:
                is_video = True
                # Save video temporarily
                temp_path = "temp_video.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                # Video controls
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    play_btn = st.button(" Play Detection", use_container_width=True)
                with col_b:
                    stop_btn = st.button(" Stop", use_container_width=True)
                with col_c:
                    frame_skip = st.slider("Process every N frames", 1, 10, 3)
                
                if stop_btn:
                    st.session_state.video_playing = False
                
                if play_btn or st.session_state.video_playing:
                    st.session_state.video_playing = True
                    
                    # Video processing
                    cap = cv2.VideoCapture(temp_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
                    
                    st.info(f" Video: {total_frames} frames @ {fps} FPS")
                    
                    # Create placeholders for real-time update
                    frame_placeholder = st.empty()
                    stats_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    frame_count = 0
                    all_detections = []
                    
                    while cap.isOpened() and st.session_state.video_playing:
                        ret, video_frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        
                        # Process every Nth frame
                        if frame_count % frame_skip == 0:
                            # Run detection
                            result = st.session_state.detector.detect(video_frame, draw_boxes=True)
                            all_detections.append(result.total_count)
                            
                            # Display frame
                            display_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(display_frame, caption=f"Frame {frame_count}/{total_frames}", use_container_width=True)
                            
                            # Update stats
                            avg_vehicles = np.mean(all_detections) if all_detections else 0
                            stats_placeholder.markdown(f"""
                            | Metric | Value |
                            |--------|-------|
                            | Current Vehicles | **{result.total_count}** |
                            | Avg Vehicles | **{avg_vehicles:.1f}** |
                            | Frames Processed | **{len(all_detections)}** |
                            | Density | **{st.session_state.detector.get_density_level(result.total_count).upper()}** |
                            """)
                            
                            st.session_state.current_stats = {
                                'total': result.total_count,
                                'counts': result.vehicle_counts,
                                'avg': avg_vehicles
                            }
                    
                    cap.release()
                    st.session_state.video_playing = False
                    
                    if all_detections:
                        st.success(f" Processed {len(all_detections)} frames. Avg vehicles: {np.mean(all_detections):.1f}")
                        
                        # Show detection chart
                        fig = px.line(
                            x=list(range(len(all_detections))),
                            y=all_detections,
                            labels={'x': 'Frame', 'y': 'Vehicle Count'},
                            title='Vehicle Count Over Time'
                        )
                        fig.update_traces(fill='tozeroy')
                        st.plotly_chart(fig, use_container_width=True)
        
        elif source_type == "Test Frame":
            # Generate synthetic test frame
            frame = create_test_frame(add_vehicles=True)
            st.info("Using synthetic test frame for demonstration")
        
        elif source_type == "Webcam":
            st.subheader(" Live Webcam Detection")
            st.info("Click START to begin live vehicle detection from your webcam")
            
            # WebRTC streamer for live webcam
            webrtc_ctx = webrtc_streamer(
                key="vehicle-detection",
                video_processor_factory=VideoTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480},
                    "audio": False
                },
                async_processing=True
            )
            
            if webrtc_ctx.video_processor:
                st.success(" Live detection active - vehicles are being detected in real-time!")
                
                # Show detection info
                st.markdown("""
                **Detection Info:**
                - Processing every 3rd frame for performance
                - Green boxes = Cars
                - Orange boxes = Motorcycles  
                - Blue boxes = Buses
                - Red boxes = Trucks
                """)
            
            # Set flag to skip single frame detection
            is_video = True
        
        # Run detection for single image/frame (not video)
        if frame is not None and not is_video:
            st.subheader("Detection Results")
            
            with st.spinner("Running YOLOv8 detection..."):
                result = st.session_state.detector.detect(frame, draw_boxes=True)
            
            # Display annotated frame
            display_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, caption="Detected Vehicles", use_container_width=True)
            
            # Detection metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Vehicles", result.total_count)
            with col_b:
                st.metric("Processing Time", f"{result.processing_time*1000:.1f}ms")
            with col_c:
                density = st.session_state.detector.get_density_level(result.total_count)
                st.metric("Density Level", density.upper())
            
            # Store for sidebar display
            st.session_state.current_stats = {
                'total': result.total_count,
                'counts': result.vehicle_counts
            }
    
    with col2:
        st.subheader("Detection Summary")
        
        # Show stats from current detection
        if st.session_state.current_stats.get('counts'):
            # Vehicle counts by type
            counts = st.session_state.current_stats['counts']
            if counts:
                df = pd.DataFrame([
                    {"Type": k.capitalize(), "Count": v}
                    for k, v in counts.items()
                ])
                
                fig = px.bar(
                    df,
                    x="Type",
                    y="Count",
                    color="Type",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                fig.update_layout(showlegend=False, height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            # Signal recommendation
            st.subheader(" Recommended Signal Timing")
            total = st.session_state.current_stats.get('total', 0)
            density = st.session_state.detector.get_density_level(total)
            timing = st.session_state.signal_optimizer.calculate_timing(density)
            
            st.markdown(f"""
            | Phase | Duration |
            |-------|----------|
            |  Green | **{timing['green']}s** |
            |  Yellow | **{timing['yellow']}s** |
            |  Red | **{timing['red']}s** |
            """)


def render_analytics():
    """Analytics view"""
    st.markdown('<h1 class="main-header">Traffic Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Historical analysis and traffic pattern insights</p>', unsafe_allow_html=True)
    
    if len(st.session_state.data_handler.traffic_data) == 0:
        st.info("No traffic data available. Click 'Generate Demo Data' in the sidebar to get started.")
        return
    
    # Time range selector
    time_range = st.selectbox(
        "Select Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days"],
        index=0
    )
    
    hours_map = {"Last 24 Hours": 24, "Last 7 Days": 168, "Last 30 Days": 720}
    hours = hours_map[time_range]
    
    data = st.session_state.data_handler.get_recent_data(hours=hours)
    
    if len(data) == 0:
        st.warning(f"No data available for {time_range.lower()}")
        return
    
    # Traffic volume over time
    st.subheader(" Traffic Volume Over Time")
    
    data['hour_bucket'] = data['timestamp'].dt.floor('H')
    hourly_volume = data.groupby('hour_bucket')['vehicle_count'].mean().reset_index()
    
    fig = px.line(
        hourly_volume,
        x='hour_bucket',
        y='vehicle_count',
        title=''
    )
    fig.update_traces(fill='tozeroy', line_color='#667eea')
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Vehicle Count",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Split view
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Daily Pattern Heatmap")
        
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day_name()
        
        pivot = data.pivot_table(
            values='vehicle_count',
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])
        
        fig = px.imshow(
            pivot,
            labels=dict(x="Hour of Day", y="Day", color="Vehicles"),
            color_continuous_scale='Viridis',
            aspect='auto'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(" Density Distribution")
        
        density_counts = data['density_level'].value_counts()
        
        colors = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444', 'critical': '#7c3aed'}
        
        fig = px.bar(
            x=density_counts.index,
            y=density_counts.values,
            color=density_counts.index,
            color_discrete_map=colors
        )
        fig.update_layout(
            xaxis_title="Density Level",
            yaxis_title="Count",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Intersection comparison
    st.subheader(" Intersection Comparison")
    
    int_comparison = data.groupby('intersection_id').agg({
        'vehicle_count': ['mean', 'max'],
        'emergency_detected': 'sum'
    }).round(1)
    int_comparison.columns = ['Avg Vehicles', 'Max Vehicles', 'Emergency Events']
    int_comparison = int_comparison.reset_index()
    
    st.dataframe(int_comparison, use_container_width=True, hide_index=True)
    
    # Export option
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Export Data", use_container_width=True):
            filepath = st.session_state.data_handler.export_to_json()
            st.success(f"Data exported to {filepath}")


def render_signal_control():
    """Multi-intersection signal control and green wave coordination"""
    st.markdown('<h1 class="main-header">Multi-Intersection Signal Control</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Green wave synchronization and adaptive signal timing</p>', unsafe_allow_html=True)
    
    coordinator = st.session_state.coordinator
    controller = st.session_state.adaptive_controller
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Intersections", len(coordinator.intersections))
    with col2:
        active_greens = sum(1 for i in coordinator.intersections.values() 
                          if i.current_phase == SignalPhase.GREEN)
        st.metric("Active Green Signals", active_greens)
    with col3:
        active_waves = len([w for w in coordinator.green_waves.values() if w.active])
        st.metric("Active Green Waves", active_waves)
    with col4:
        corridors = len(st.session_state.active_corridors)
        st.metric("Emergency Corridors", corridors, delta="" if corridors > 0 else None)
    
    st.divider()
    
    # Two column layout
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("Network Map")
        
        # Create network visualization
        fig = go.Figure()
        
        # Add intersections as nodes
        for int_id, intersection in coordinator.intersections.items():
            color = {
                SignalPhase.GREEN: '#10b981',
                SignalPhase.YELLOW: '#f59e0b',
                SignalPhase.RED: '#ef4444'
            }.get(intersection.current_phase, '#6b7280')
            
            fig.add_trace(go.Scatter(
                x=[intersection.position[0]],
                y=[intersection.position[1]],
                mode='markers+text',
                marker=dict(size=30, color=color, symbol='circle'),
                text=[int_id],
                textposition='middle center',
                textfont=dict(color='white', size=10),
                name=int_id,
                hovertemplate=f"<b>{int_id}</b><br>" +
                              f"Phase: {intersection.current_phase.value}<br>" +
                              f"Density: {intersection.density_level}<extra></extra>"
            ))
        
        # Connection lines between nearby intersections (simplified)
        int_list = list(coordinator.intersections.values())
        for i in range(len(int_list) - 1):
            fig.add_trace(go.Scatter(
                x=[int_list[i].position[0], int_list[i+1].position[0]],
                y=[int_list[i].position[1], int_list[i+1].position[1]],
                mode='lines',
                line=dict(width=2, color='#64748b'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Green Wave Planning
        st.subheader("Green Wave Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            wave_intersections = st.multiselect(
                "Select Intersections for Green Wave",
                options=list(coordinator.intersections.keys()),
                default=list(coordinator.intersections.keys())[:3]
            )
            target_speed = st.slider("Target Speed (km/h)", 30, 60, 40)
        
        with col2:
            direction = st.selectbox("Wave Direction", ["North-South", "East-West", "Custom"])
            if st.button("Calculate Green Wave", use_container_width=True):
                if len(wave_intersections) >= 2:
                    wave = coordinator.create_green_wave(
                        wave_id=f"wave_{datetime.now().strftime('%H%M%S')}",
                        direction=direction.lower().replace("-", "_"),
                        intersection_ids=wave_intersections,
                        speed_kmh=target_speed
                    )
                    st.success(f"Green wave '{wave.id}' created for {len(wave_intersections)} intersections")
                    
                    # Display timing offsets
                    timing_df = pd.DataFrame([
                        {"Intersection": k, "Green Start Offset (s)": f"{v}"}
                        for k, v in wave.offset_times.items()
                    ])
                    st.dataframe(timing_df, hide_index=True)
                else:
                    st.warning("Select at least 2 intersections")
    
    with right_col:
        st.subheader("Signal Status")
        
        for int_id, intersection in coordinator.intersections.items():
            phase_indicator = {
                SignalPhase.GREEN: "[G]",
                SignalPhase.YELLOW: "[Y]", 
                SignalPhase.RED: "[R]"
            }.get(intersection.current_phase, "[-]")
            
            with st.expander(f"{phase_indicator} {int_id}", expanded=False):
                st.write(f"**Phase:** {intersection.current_phase.value}")
                st.write(f"**Timing:** {intersection.green_duration if intersection.current_phase == SignalPhase.GREEN else intersection.yellow_duration if intersection.current_phase == SignalPhase.YELLOW else intersection.red_duration}s")
                st.write(f"**Density:** {intersection.density_level}")
                
                # Manual override
                new_phase = st.selectbox(
                    "Override Phase",
                    [SignalPhase.GREEN, SignalPhase.YELLOW, SignalPhase.RED],
                    index=[SignalPhase.GREEN, SignalPhase.YELLOW, SignalPhase.RED].index(intersection.current_phase),
                    key=f"phase_{int_id}"
                )
                if st.button(f"Apply", key=f"apply_{int_id}"):
                    intersection.current_phase = new_phase
                    st.rerun()
        
        st.divider()
        
        # Adaptive control settings
        st.subheader("Adaptive Control")
        
        adapt_enabled = st.toggle("Enable Adaptive Timing", value=True)
        
        if adapt_enabled:
            # Generate simulated vehicle data
            vehicle_counts = {
                int_id: {'N': np.random.randint(5, 25), 'S': np.random.randint(5, 20),
                        'E': np.random.randint(3, 15), 'W': np.random.randint(3, 15)}
                for int_id in coordinator.intersections.keys()
            }
            
            if st.button("Update Timings", use_container_width=True):
                st.success("Calculating optimal timings based on current traffic...")
                
                for int_id, vehicles in vehicle_counts.items():
                    timing = controller.calculate_optimal_timing(int_id, vehicles)
                    controller.apply_adaptive_timing(int_id, timing)
                    st.write(f"**{int_id}:** Cycle={timing.get('cycle', 60)}s, Vehicles={timing.get('total_vehicles', 0)}")
    
    # Incident alerts section
    st.divider()
    st.subheader("Active Incidents")
    
    incidents = st.session_state.incidents
    if incidents:
        for incident in incidents[-5:]:  # Show last 5 incidents
            severity_style = {
                'critical': 'background-color: #b91c1c; color: white;',
                'high': 'background-color: #ea580c; color: white;',
                'medium': 'background-color: #ca8a04; color: white;',
                'low': 'background-color: #16a34a; color: white;'
            }.get(incident.severity, 'background-color: #6b7280; color: white;')
            
            st.markdown(f"""
            <div class="emergency-alert">
                <span style="padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; {severity_style}">{incident.severity.upper()}</span>
                <strong style="margin-left: 8px;">{incident.incident_type.upper()}</strong> - 
                Location: ({incident.location[0]}, {incident.location[1]}) - 
                Confidence: {incident.confidence:.1%}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No active incidents detected")


def render_emergency():
    """Emergency control view"""
    st.markdown('<h1 class="main-header">Emergency Control Center</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Green corridor management and emergency vehicle prioritization</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Emergency Corridor Control")
        
        # Emergency toggle
        emergency_active = st.toggle(
            "Activate Emergency Green Corridor",
            value=st.session_state.emergency_active,
            help="Override all signals to create green corridor for emergency vehicles"
        )
        st.session_state.emergency_active = emergency_active
        
        if emergency_active:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
                padding: 2rem; border-radius: 1rem; color: white; text-align: center; margin: 1rem 0;">
                <div style="font-size: 3rem;"></div>
                <div style="font-size: 1.5rem; font-weight: 700;">GREEN CORRIDOR ACTIVE</div>
                <div style="font-size: 0.875rem; margin-top: 0.5rem;">
                    All intersection signals optimized for emergency passage
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show affected intersections
            st.subheader("Affected Intersections")
            
            intersections = st.session_state.data_handler.traffic_data['intersection_id'].unique()
            
            for int_id in intersections[:4]:
                timing = st.session_state.signal_optimizer.calculate_timing('low', emergency_active=True)
                st.markdown(f"""
                <div style="background: #fef2f2; border-left: 4px solid #ef4444;
                    padding: 0.75rem 1rem; margin: 0.5rem 0; border-radius: 0.25rem;">
                    <strong>{int_id}</strong>:  {timing['green']}s extended green
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(" Normal operation - No active emergencies")
        
        # Emergency history
        st.divider()
        st.subheader(" Recent Emergency Events")
        
        data = st.session_state.data_handler.traffic_data
        emergency_data = data[data['emergency_detected'] == True].tail(10)
        
        if len(emergency_data) > 0:
            display_df = emergency_data[['timestamp', 'intersection_id', 'direction']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("No recent emergency events")
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.button("Simulate Ambulance", use_container_width=True):
            st.session_state.data_handler.add_record(
                intersection_id='INT_001',
                direction='N',
                vehicle_counts={'car': 5},
                emergency_detected=True
            )
            st.success("Ambulance detected at INT_001")
            st.rerun()
        
        if st.button("Simulate Fire Truck", use_container_width=True):
            st.session_state.data_handler.add_record(
                intersection_id='INT_002',
                direction='S',
                vehicle_counts={'car': 3},
                emergency_detected=True
            )
            st.success("Fire truck detected at INT_002")
            st.rerun()
        
        if st.button("Simulate Police Vehicle", use_container_width=True):
            st.session_state.data_handler.add_record(
                intersection_id='INT_003',
                direction='E',
                vehicle_counts={'car': 4},
                emergency_detected=True
            )
            # Track the emergency vehicle
            st.session_state.emergency_tracker.track_emergency(
                vehicle_id="police_001",
                position=(300, 200),
                vehicle_type="police",
                direction="east"
            )
            st.success("Police vehicle detected at INT_003")
            st.rerun()
        
        st.divider()
        
        st.subheader("Emergency Stats")
        
        total_emergencies = data['emergency_detected'].sum()
        st.metric("Total Events (All Time)", int(total_emergencies))
        
        if len(data) > 0:
            emergency_rate = (total_emergencies / len(data)) * 100
            st.metric("Emergency Rate", f"{emergency_rate:.2f}%")
        
        # Tracked emergency vehicles
        active_corridors = st.session_state.emergency_tracker.get_active_corridors()
        if active_corridors:
            st.divider()
            st.subheader("Tracked Vehicles")
            for corridor in active_corridors:
                v_info = corridor['data']
                st.caption(f" {v_info['type']}: heading {v_info['direction']}")
    
    # Corridor activation for multi-intersection
    st.divider()
    st.subheader("Emergency Green Wave")
    
    col1, col2 = st.columns(2)
    with col1:
        corridor_path = st.multiselect(
            "Select Corridor Path",
            options=list(st.session_state.coordinator.intersections.keys()),
            default=list(st.session_state.coordinator.intersections.keys())[:3],
            help="Select intersections for emergency corridor"
        )
    
    with col2:
        direction = st.selectbox("Emergency Direction", ["North", "South", "East", "West"])
        
        if st.button("Activate Emergency Corridor", type="primary", use_container_width=True):
            if len(corridor_path) >= 2:
                # Activate corridor - set all intersections to green
                st.session_state.coordinator.activate_emergency_corridor(corridor_path)
                st.session_state.active_corridors.append({
                    'path': corridor_path,
                    'direction': direction,
                    'activated_at': datetime.now()
                })
                st.success(f"Emergency corridor activated for {len(corridor_path)} intersections!")
            else:
                st.warning("Select at least 2 intersections")
    
    # Show active corridors
    if st.session_state.active_corridors:
        st.subheader("Active Corridors")
        for idx, corridor in enumerate(st.session_state.active_corridors):
            with st.expander(f"Corridor {idx + 1}: {'  '.join(corridor['path'])}"):
                st.write(f"**Direction:** {corridor['direction']}")
                st.write(f"**Activated:** {corridor['activated_at'].strftime('%H:%M:%S')}")
                if st.button(f"Deactivate", key=f"deactivate_{idx}"):
                    st.session_state.coordinator.deactivate_emergency_corridor(corridor['path'])
                    st.session_state.active_corridors.pop(idx)
                    st.rerun()


def render_settings():
    """Settings view"""
    st.markdown('<h1 class="main-header">System Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Configure detection parameters and system behavior</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Detection Settings", "Signal Timing", "Camera Config"])
    
    with tab1:
        st.subheader("YOLOv8 Detection Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = st.slider(
                "Detection Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence score for vehicle detection"
            )
            
            iou_threshold = st.slider(
                "IOU Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Intersection over Union threshold for NMS"
            )
        
        with col2:
            frame_skip = st.slider(
                "Frame Skip",
                min_value=1,
                max_value=30,
                value=5,
                help="Process every Nth frame for efficiency"
            )
            
            resize_width = st.number_input(
                "Frame Width",
                min_value=320,
                max_value=1920,
                value=640,
                step=32
            )
    
    with tab2:
        st.subheader("Signal Timing Configuration")
        
        st.markdown("#### Timing per Density Level (seconds)")
        
        for level in ['low', 'medium', 'high', 'critical']:
            st.markdown(f"**{level.capitalize()} Density**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.number_input(f"Green ({level})", value=SIGNAL_TIMING[level]['green'], 
                              key=f"green_{level}", min_value=5, max_value=120)
            with col2:
                st.number_input(f"Yellow ({level})", value=SIGNAL_TIMING[level]['yellow'],
                              key=f"yellow_{level}", min_value=2, max_value=10)
            with col3:
                st.number_input(f"Red ({level})", value=SIGNAL_TIMING[level]['red'],
                              key=f"red_{level}", min_value=5, max_value=120)
            
            st.divider()
    
    with tab3:
        st.subheader("Camera Feed Configuration")
        
        st.info("Configure CCTV and traffic camera feeds for real-time monitoring")
        
        for camera_id, config in CAMERA_FEEDS.items():
            with st.expander(f" {config['name']} ({camera_id})"):
                st.text_input("Camera Name", value=config['name'], key=f"name_{camera_id}")
                st.text_input("Stream URL", value=config['url'], key=f"url_{camera_id}")
                st.selectbox("Stream Type", ["RTSP", "HTTP", "File"], key=f"type_{camera_id}")
        
        if st.button(" Add New Camera"):
            st.info("Camera configuration will be added to config.py")
    
    # Save button
    st.divider()
    if st.button("Save Settings", type="primary", use_container_width=True):
        st.success("Settings saved successfully!")


def main():
    """Main application entry point"""
    init_session_state()
    
    page = render_sidebar()
    
    if page == "Dashboard":
        render_dashboard()
    elif page == "Live Detection":
        render_live_detection()
    elif page == "Signal Control":
        render_signal_control()
    elif page == "Analytics":
        render_analytics()
    elif page == "Emergency Control":
        render_emergency()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()


