"""
Real-Time Healthcare Simulation Engine
Advanced real-time simulation system with live parameter adjustment, instant feedback,
interactive scenario modeling, and dynamic visualization updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import threading
import time
import json
import logging
from datetime import datetime, timedelta
import queue
import websockets
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SimulationState(Enum):
    """Real-time simulation states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class ParameterType(Enum):
    """Types of simulation parameters"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"

class UpdateFrequency(Enum):
    """Update frequency options"""
    REAL_TIME = 0.1  # 100ms
    FAST = 0.5       # 500ms
    NORMAL = 1.0     # 1 second
    SLOW = 2.0       # 2 seconds

@dataclass
class SimulationParameter:
    """Real-time adjustable simulation parameter"""
    name: str
    display_name: str
    param_type: ParameterType
    current_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None
    options: Optional[List[Any]] = None
    description: str = ""
    unit: str = ""
    category: str = "General"

@dataclass
class SimulationMetric:
    """Real-time simulation output metric"""
    name: str
    display_name: str
    current_value: float
    history: List[float] = field(default_factory=list)
    target_range: Optional[Tuple[float, float]] = None
    unit: str = ""
    format_string: str = "{:.2f}"
    color: str = "#1f77b4"

@dataclass
class SimulationEvent:
    """Real-time simulation event"""
    timestamp: datetime
    event_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    impact: str = "low"  # low, medium, high

class RealTimeSimulationEngine:
    """Advanced real-time simulation engine for healthcare analytics"""
    
    def __init__(self, update_frequency: UpdateFrequency = UpdateFrequency.NORMAL):
        self.state = SimulationState.STOPPED
        self.update_frequency = update_frequency.value
        self.parameters = {}
        self.metrics = {}
        self.events = []
        self.simulation_thread = None
        self.stop_event = threading.Event()
        self.parameter_queue = queue.Queue()
        self.callback_functions = {}
        self.data_history = []
        self.max_history_length = 1000
        
        # WebSocket for real-time updates
        self.websocket_clients = set()
        self.websocket_server = None
        
        # Initialize default healthcare simulation
        self._initialize_healthcare_simulation()
        
        # Dash app for real-time dashboard
        self.app = None
        self._setup_dashboard()
    
    def _initialize_healthcare_simulation(self):
        """Initialize default healthcare simulation parameters and metrics"""
        
        # Healthcare simulation parameters
        self.add_parameter(SimulationParameter(
            name="patient_arrival_rate",
            display_name="Patient Arrival Rate",
            param_type=ParameterType.CONTINUOUS,
            current_value=50.0,
            min_value=10.0,
            max_value=200.0,
            step_size=5.0,
            description="Number of patients arriving per hour",
            unit="patients/hour",
            category="Operations"
        ))
        
        self.add_parameter(SimulationParameter(
            name="staff_efficiency",
            display_name="Staff Efficiency",
            param_type=ParameterType.CONTINUOUS,
            current_value=0.85,
            min_value=0.5,
            max_value=1.0,
            step_size=0.05,
            description="Staff efficiency factor (0-1)",
            unit="efficiency",
            category="Staffing"
        ))
        
        self.add_parameter(SimulationParameter(
            name="bed_capacity",
            display_name="Bed Capacity",
            param_type=ParameterType.DISCRETE,
            current_value=100,
            min_value=50,
            max_value=300,
            step_size=10,
            description="Total number of available beds",
            unit="beds",
            category="Resources"
        ))
        
        self.add_parameter(SimulationParameter(
            name="emergency_mode",
            display_name="Emergency Mode",
            param_type=ParameterType.BOOLEAN,
            current_value=False,
            description="Enable emergency protocols",
            category="Operations"
        ))
        
        self.add_parameter(SimulationParameter(
            name="season",
            display_name="Season",
            param_type=ParameterType.CATEGORICAL,
            current_value="normal",
            options=["normal", "flu_season", "pandemic", "holiday"],
            description="Current seasonal conditions",
            category="Environment"
        ))
        
        # Healthcare metrics
        self.add_metric(SimulationMetric(
            name="wait_time",
            display_name="Average Wait Time",
            current_value=0.0,
            target_range=(0, 30),
            unit="minutes",
            color="#ff7f0e"
        ))
        
        self.add_metric(SimulationMetric(
            name="bed_occupancy",
            display_name="Bed Occupancy Rate",
            current_value=0.0,
            target_range=(0.7, 0.9),
            unit="%",
            format_string="{:.1f}%",
            color="#2ca02c"
        ))
        
        self.add_metric(SimulationMetric(
            name="patient_satisfaction",
            display_name="Patient Satisfaction",
            current_value=0.0,
            target_range=(8.0, 10.0),
            unit="score",
            format_string="{:.1f}/10",
            color="#d62728"
        ))
        
        self.add_metric(SimulationMetric(
            name="staff_utilization",
            display_name="Staff Utilization",
            current_value=0.0,
            target_range=(0.6, 0.8),
            unit="%",
            format_string="{:.1f}%",
            color="#9467bd"
        ))
        
        self.add_metric(SimulationMetric(
            name="cost_per_patient",
            display_name="Cost per Patient",
            current_value=0.0,
            target_range=(800, 1200),
            unit="$",
            format_string="${:.0f}",
            color="#8c564b"
        ))
    
    def add_parameter(self, parameter: SimulationParameter):
        """Add a simulation parameter"""
        self.parameters[parameter.name] = parameter
    
    def add_metric(self, metric: SimulationMetric):
        """Add a simulation metric"""
        self.metrics[metric.name] = metric
    
    def update_parameter(self, name: str, value: Any):
        """Update a parameter value in real-time"""
        if name in self.parameters:
            old_value = self.parameters[name].current_value
            self.parameters[name].current_value = value
            
            # Add to parameter queue for processing
            self.parameter_queue.put({
                'name': name,
                'old_value': old_value,
                'new_value': value,
                'timestamp': datetime.now()
            })
            
            # Log event
            self.add_event(SimulationEvent(
                timestamp=datetime.now(),
                event_type="parameter_change",
                description=f"{self.parameters[name].display_name} changed from {old_value} to {value}",
                parameters={'parameter': name, 'old_value': old_value, 'new_value': value}
            ))
            
            # Trigger callbacks
            if name in self.callback_functions:
                for callback in self.callback_functions[name]:
                    try:
                        callback(name, old_value, value)
                    except Exception as e:
                        logger.error(f"Error in parameter callback: {e}")
    
    def add_event(self, event: SimulationEvent):
        """Add a simulation event"""
        self.events.append(event)
        
        # Keep only recent events
        if len(self.events) > 100:
            self.events = self.events[-100:]
    
    def register_parameter_callback(self, parameter_name: str, callback: Callable):
        """Register a callback for parameter changes"""
        if parameter_name not in self.callback_functions:
            self.callback_functions[parameter_name] = []
        self.callback_functions[parameter_name].append(callback)
    
    def start_simulation(self):
        """Start the real-time simulation"""
        if self.state == SimulationState.RUNNING:
            return
        
        self.state = SimulationState.RUNNING
        self.stop_event.clear()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        
        # Start WebSocket server
        asyncio.create_task(self._start_websocket_server())
        
        self.add_event(SimulationEvent(
            timestamp=datetime.now(),
            event_type="simulation_start",
            description="Real-time simulation started"
        ))
    
    def stop_simulation(self):
        """Stop the real-time simulation"""
        self.state = SimulationState.STOPPED
        self.stop_event.set()
        
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=2.0)
        
        self.add_event(SimulationEvent(
            timestamp=datetime.now(),
            event_type="simulation_stop",
            description="Real-time simulation stopped"
        ))
    
    def pause_simulation(self):
        """Pause the real-time simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            
            self.add_event(SimulationEvent(
                timestamp=datetime.now(),
                event_type="simulation_pause",
                description="Real-time simulation paused"
            ))
    
    def resume_simulation(self):
        """Resume the real-time simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            
            self.add_event(SimulationEvent(
                timestamp=datetime.now(),
                event_type="simulation_resume",
                description="Real-time simulation resumed"
            ))
    
    def _simulation_loop(self):
        """Main simulation loop running in separate thread"""
        
        iteration = 0
        
        while not self.stop_event.is_set():
            try:
                if self.state == SimulationState.RUNNING:
                    
                    # Process parameter changes
                    self._process_parameter_changes()
                    
                    # Update simulation metrics
                    self._update_metrics(iteration)
                    
                    # Store data point
                    self._store_data_point()
                    
                    # Broadcast updates to WebSocket clients
                    asyncio.create_task(self._broadcast_updates())
                    
                    iteration += 1
                
                # Sleep based on update frequency
                time.sleep(self.update_frequency)
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                self.state = SimulationState.ERROR
                self.add_event(SimulationEvent(
                    timestamp=datetime.now(),
                    event_type="simulation_error",
                    description=f"Simulation error: {str(e)}",
                    impact="high"
                ))
    
    def _process_parameter_changes(self):
        """Process queued parameter changes"""
        
        while not self.parameter_queue.empty():
            try:
                change = self.parameter_queue.get_nowait()
                # Parameter changes are already processed in update_parameter
                # This could be used for additional processing if needed
            except queue.Empty:
                break
    
    def _update_metrics(self, iteration: int):
        """Update simulation metrics based on current parameters"""
        
        # Get current parameter values
        arrival_rate = self.parameters["patient_arrival_rate"].current_value
        staff_efficiency = self.parameters["staff_efficiency"].current_value
        bed_capacity = self.parameters["bed_capacity"].current_value
        emergency_mode = self.parameters["emergency_mode"].current_value
        season = self.parameters["season"].current_value
        
        # Seasonal multipliers
        seasonal_multipliers = {
            "normal": 1.0,
            "flu_season": 1.3,
            "pandemic": 2.0,
            "holiday": 0.8
        }
        
        seasonal_factor = seasonal_multipliers.get(season, 1.0)
        effective_arrival_rate = arrival_rate * seasonal_factor
        
        # Emergency mode effects
        if emergency_mode:
            staff_efficiency *= 1.2  # 20% efficiency boost
            effective_arrival_rate *= 1.5  # 50% more patients
        
        # Calculate wait time (simplified model)
        base_wait_time = max(0, (effective_arrival_rate - bed_capacity * 0.8) / staff_efficiency)
        wait_time = base_wait_time + np.random.normal(0, 5)  # Add noise
        wait_time = max(0, wait_time)
        
        # Calculate bed occupancy
        occupancy_rate = min(0.95, effective_arrival_rate / bed_capacity)
        occupancy_rate += np.random.normal(0, 0.05)  # Add noise
        occupancy_rate = max(0, min(1, occupancy_rate))
        
        # Calculate patient satisfaction (inversely related to wait time)
        satisfaction = 10 - (wait_time / 10)  # Decreases with wait time
        satisfaction *= staff_efficiency  # Increases with staff efficiency
        satisfaction += np.random.normal(0, 0.5)  # Add noise
        satisfaction = max(0, min(10, satisfaction))
        
        # Calculate staff utilization
        utilization = min(0.95, effective_arrival_rate / (bed_capacity * 1.2))
        utilization *= (2 - staff_efficiency)  # Higher when efficiency is lower
        utilization += np.random.normal(0, 0.05)  # Add noise
        utilization = max(0, min(1, utilization))
        
        # Calculate cost per patient
        base_cost = 1000
        cost_per_patient = base_cost * (1 + wait_time / 100)  # Increases with wait time
        cost_per_patient *= (2 - staff_efficiency)  # Increases when efficiency is lower
        if emergency_mode:
            cost_per_patient *= 1.3  # 30% increase in emergency mode
        cost_per_patient += np.random.normal(0, 50)  # Add noise
        cost_per_patient = max(500, cost_per_patient)
        
        # Update metrics
        self._update_metric("wait_time", wait_time)
        self._update_metric("bed_occupancy", occupancy_rate * 100)
        self._update_metric("patient_satisfaction", satisfaction)
        self._update_metric("staff_utilization", utilization * 100)
        self._update_metric("cost_per_patient", cost_per_patient)
    
    def _update_metric(self, name: str, value: float):
        """Update a specific metric"""
        if name in self.metrics:
            self.metrics[name].current_value = value
            self.metrics[name].history.append(value)
            
            # Keep history within limits
            if len(self.metrics[name].history) > self.max_history_length:
                self.metrics[name].history = self.metrics[name].history[-self.max_history_length:]
    
    def _store_data_point(self):
        """Store current simulation state as data point"""
        
        data_point = {
            'timestamp': datetime.now(),
            'parameters': {name: param.current_value for name, param in self.parameters.items()},
            'metrics': {name: metric.current_value for name, metric in self.metrics.items()}
        }
        
        self.data_history.append(data_point)
        
        # Keep history within limits
        if len(self.data_history) > self.max_history_length:
            self.data_history = self.data_history[-self.max_history_length:]
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
        
        try:
            self.websocket_server = await websockets.serve(handle_client, "localhost", 8765)
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _broadcast_updates(self):
        """Broadcast updates to WebSocket clients"""
        
        if not self.websocket_clients:
            return
        
        update_data = {
            'timestamp': datetime.now().isoformat(),
            'state': self.state.value,
            'parameters': {
                name: {
                    'value': param.current_value,
                    'display_name': param.display_name,
                    'unit': param.unit
                }
                for name, param in self.parameters.items()
            },
            'metrics': {
                name: {
                    'value': metric.current_value,
                    'display_name': metric.display_name,
                    'unit': metric.unit,
                    'format_string': metric.format_string,
                    'target_range': metric.target_range
                }
                for name, metric in self.metrics.items()
            },
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'type': event.event_type,
                    'description': event.description,
                    'impact': event.impact
                }
                for event in self.events[-5:]  # Last 5 events
            ]
        }
        
        message = json.dumps(update_data)
        
        # Send to all connected clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def _setup_dashboard(self):
        """Setup Dash dashboard for real-time visualization"""
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Dashboard layout
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Real-Time Healthcare Simulation Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulation Controls"),
                        dbc.CardBody([
                            dbc.ButtonGroup([
                                dbc.Button("Start", id="start-btn", color="success", className="me-2"),
                                dbc.Button("Pause", id="pause-btn", color="warning", className="me-2"),
                                dbc.Button("Stop", id="stop-btn", color="danger")
                            ]),
                            html.Hr(),
                            html.Div(id="simulation-status", className="mt-2")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Parameter Controls
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Real-Time Parameter Controls"),
                        dbc.CardBody([
                            html.Div(id="parameter-controls")
                        ])
                    ])
                ], width=6),
                
                # Real-Time Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Real-Time Metrics"),
                        dbc.CardBody([
                            html.Div(id="metrics-display")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Real-Time Charts
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id="realtime-chart")
                ], width=12)
            ], className="mb-4"),
            
            # Events Log
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Events"),
                        dbc.CardBody([
                            html.Div(id="events-log")
                        ])
                    ])
                ], width=12)
            ]),
            
            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every second
                n_intervals=0
            )
        ], fluid=True)
        
        # Callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for interactivity"""
        
        @self.app.callback(
            [Output('simulation-status', 'children'),
             Output('parameter-controls', 'children'),
             Output('metrics-display', 'children'),
             Output('realtime-chart', 'figure'),
             Output('events-log', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('start-btn', 'n_clicks'),
             Input('pause-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks')]
        )
        def update_dashboard(n_intervals, start_clicks, pause_clicks, stop_clicks):
            
            # Handle button clicks
            ctx = dash.callback_context
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                
                if button_id == 'start-btn':
                    self.start_simulation()
                elif button_id == 'pause-btn':
                    if self.state == SimulationState.RUNNING:
                        self.pause_simulation()
                    else:
                        self.resume_simulation()
                elif button_id == 'stop-btn':
                    self.stop_simulation()
            
            # Update status
            status_color = {
                SimulationState.STOPPED: "secondary",
                SimulationState.RUNNING: "success",
                SimulationState.PAUSED: "warning",
                SimulationState.ERROR: "danger"
            }
            
            status = dbc.Badge(
                f"Status: {self.state.value.title()}",
                color=status_color[self.state],
                className="fs-6"
            )
            
            # Generate parameter controls
            parameter_controls = []
            for name, param in self.parameters.items():
                
                if param.param_type == ParameterType.CONTINUOUS:
                    control = dbc.Row([
                        dbc.Col([
                            dbc.Label(f"{param.display_name} ({param.unit})"),
                            dcc.Slider(
                                id=f"slider-{name}",
                                min=param.min_value,
                                max=param.max_value,
                                step=param.step_size,
                                value=param.current_value,
                                marks={
                                    param.min_value: str(param.min_value),
                                    param.max_value: str(param.max_value)
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ], className="mb-3")
                
                elif param.param_type == ParameterType.BOOLEAN:
                    control = dbc.Row([
                        dbc.Col([
                            dbc.Switch(
                                id=f"switch-{name}",
                                label=param.display_name,
                                value=param.current_value
                            )
                        ])
                    ], className="mb-3")
                
                elif param.param_type == ParameterType.CATEGORICAL:
                    control = dbc.Row([
                        dbc.Col([
                            dbc.Label(param.display_name),
                            dcc.Dropdown(
                                id=f"dropdown-{name}",
                                options=[{'label': opt, 'value': opt} for opt in param.options],
                                value=param.current_value
                            )
                        ])
                    ], className="mb-3")
                
                else:  # DISCRETE
                    control = dbc.Row([
                        dbc.Col([
                            dbc.Label(f"{param.display_name} ({param.unit})"),
                            dbc.Input(
                                id=f"input-{name}",
                                type="number",
                                min=param.min_value,
                                max=param.max_value,
                                step=param.step_size,
                                value=param.current_value
                            )
                        ])
                    ], className="mb-3")
                
                parameter_controls.append(control)
            
            # Generate metrics display
            metrics_display = []
            for name, metric in self.metrics.items():
                
                # Determine color based on target range
                color = "primary"
                if metric.target_range:
                    if metric.current_value < metric.target_range[0]:
                        color = "warning"
                    elif metric.current_value > metric.target_range[1]:
                        color = "danger"
                    else:
                        color = "success"
                
                formatted_value = metric.format_string.format(metric.current_value)
                
                metric_card = dbc.Card([
                    dbc.CardBody([
                        html.H5(metric.display_name, className="card-title"),
                        html.H3(formatted_value, className=f"text-{color}"),
                        html.Small(f"Target: {metric.target_range}" if metric.target_range else "")
                    ])
                ], className="mb-2")
                
                metrics_display.append(metric_card)
            
            # Generate real-time chart
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=list(self.metrics.keys())[:4],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for i, (name, metric) in enumerate(list(self.metrics.items())[:4]):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                if len(metric.history) > 0:
                    x_data = list(range(len(metric.history)))
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=metric.history,
                            mode='lines',
                            name=metric.display_name,
                            line=dict(color=metric.color)
                        ),
                        row=row, col=col
                    )
                    
                    # Add target range if available
                    if metric.target_range:
                        fig.add_hline(
                            y=metric.target_range[0],
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5,
                            row=row, col=col
                        )
                        fig.add_hline(
                            y=metric.target_range[1],
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5,
                            row=row, col=col
                        )
            
            fig.update_layout(
                height=600,
                title_text="Real-Time Metrics Dashboard",
                showlegend=False
            )
            
            # Generate events log
            events_log = []
            for event in self.events[-10:]:  # Last 10 events
                
                color = {
                    "low": "light",
                    "medium": "warning",
                    "high": "danger"
                }.get(event.impact, "light")
                
                event_item = dbc.ListGroupItem([
                    html.Div([
                        html.Strong(event.event_type.replace('_', ' ').title()),
                        html.Small(f" - {event.timestamp.strftime('%H:%M:%S')}", className="text-muted")
                    ]),
                    html.P(event.description, className="mb-0")
                ], color=color)
                
                events_log.append(event_item)
            
            events_log_component = dbc.ListGroup(events_log) if events_log else html.P("No events yet")
            
            return status, parameter_controls, metrics_display, fig, events_log_component
        
        # Add callbacks for parameter updates
        for name, param in self.parameters.items():
            
            if param.param_type == ParameterType.CONTINUOUS:
                @self.app.callback(
                    Output(f"slider-{name}", "value"),
                    [Input(f"slider-{name}", "value")],
                    prevent_initial_call=True
                )
                def update_continuous_param(value, param_name=name):
                    self.update_parameter(param_name, value)
                    return value
            
            elif param.param_type == ParameterType.BOOLEAN:
                @self.app.callback(
                    Output(f"switch-{name}", "value"),
                    [Input(f"switch-{name}", "value")],
                    prevent_initial_call=True
                )
                def update_boolean_param(value, param_name=name):
                    self.update_parameter(param_name, value)
                    return value
            
            elif param.param_type == ParameterType.CATEGORICAL:
                @self.app.callback(
                    Output(f"dropdown-{name}", "value"),
                    [Input(f"dropdown-{name}", "value")],
                    prevent_initial_call=True
                )
                def update_categorical_param(value, param_name=name):
                    self.update_parameter(param_name, value)
                    return value
            
            else:  # DISCRETE
                @self.app.callback(
                    Output(f"input-{name}", "value"),
                    [Input(f"input-{name}", "value")],
                    prevent_initial_call=True
                )
                def update_discrete_param(value, param_name=name):
                    if value is not None:
                        self.update_parameter(param_name, value)
                    return value
    
    def run_dashboard(self, host="127.0.0.1", port=8051, debug=False):
        """Run the real-time dashboard"""
        
        if self.app:
            self.app.run_server(host=host, port=port, debug=debug)
        else:
            logger.error("Dashboard not initialized")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        
        return {
            'state': self.state.value,
            'parameters': {
                name: {
                    'value': param.current_value,
                    'display_name': param.display_name,
                    'type': param.param_type.value,
                    'unit': param.unit,
                    'category': param.category
                }
                for name, param in self.parameters.items()
            },
            'metrics': {
                name: {
                    'value': metric.current_value,
                    'display_name': metric.display_name,
                    'unit': metric.unit,
                    'target_range': metric.target_range,
                    'history_length': len(metric.history)
                }
                for name, metric in self.metrics.items()
            },
            'events_count': len(self.events),
            'data_history_length': len(self.data_history)
        }
    
    def export_data(self, format_type: str = "json") -> Union[str, pd.DataFrame]:
        """Export simulation data"""
        
        if format_type == "json":
            return json.dumps({
                'parameters': {name: param.current_value for name, param in self.parameters.items()},
                'metrics_history': {name: metric.history for name, metric in self.metrics.items()},
                'events': [
                    {
                        'timestamp': event.timestamp.isoformat(),
                        'type': event.event_type,
                        'description': event.description,
                        'impact': event.impact
                    }
                    for event in self.events
                ],
                'data_history': [
                    {
                        'timestamp': dp['timestamp'].isoformat(),
                        'parameters': dp['parameters'],
                        'metrics': dp['metrics']
                    }
                    for dp in self.data_history
                ]
            }, indent=2)
        
        elif format_type == "dataframe":
            # Convert to DataFrame
            data_rows = []
            for dp in self.data_history:
                row = {'timestamp': dp['timestamp']}
                row.update({f"param_{k}": v for k, v in dp['parameters'].items()})
                row.update({f"metric_{k}": v for k, v in dp['metrics'].items()})
                data_rows.append(row)
            
            return pd.DataFrame(data_rows)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")

# Example usage and scenario presets
class HealthcareScenarios:
    """Predefined healthcare simulation scenarios"""
    
    @staticmethod
    def normal_operations() -> Dict[str, Any]:
        """Normal hospital operations scenario"""
        return {
            "patient_arrival_rate": 50.0,
            "staff_efficiency": 0.85,
            "bed_capacity": 100,
            "emergency_mode": False,
            "season": "normal"
        }
    
    @staticmethod
    def flu_season() -> Dict[str, Any]:
        """Flu season scenario with increased patient load"""
        return {
            "patient_arrival_rate": 75.0,
            "staff_efficiency": 0.80,
            "bed_capacity": 100,
            "emergency_mode": False,
            "season": "flu_season"
        }
    
    @staticmethod
    def pandemic_response() -> Dict[str, Any]:
        """Pandemic response scenario"""
        return {
            "patient_arrival_rate": 120.0,
            "staff_efficiency": 0.75,
            "bed_capacity": 150,
            "emergency_mode": True,
            "season": "pandemic"
        }
    
    @staticmethod
    def staff_shortage() -> Dict[str, Any]:
        """Staff shortage scenario"""
        return {
            "patient_arrival_rate": 60.0,
            "staff_efficiency": 0.60,
            "bed_capacity": 100,
            "emergency_mode": False,
            "season": "normal"
        }

# Example usage
if __name__ == "__main__":
    
    # Create real-time simulation engine
    engine = RealTimeSimulationEngine(update_frequency=UpdateFrequency.NORMAL)
    
    # Add custom parameter callback
    def on_arrival_rate_change(param_name, old_value, new_value):
        print(f"Patient arrival rate changed from {old_value} to {new_value}")
        if new_value > 100:
            print("WARNING: High patient arrival rate detected!")
    
    engine.register_parameter_callback("patient_arrival_rate", on_arrival_rate_change)
    
    # Load a scenario
    scenario = HealthcareScenarios.pandemic_response()
    for param_name, value in scenario.items():
        engine.update_parameter(param_name, value)
    
    print("Real-Time Healthcare Simulation Engine")
    print("=====================================")
    print(f"Current state: {engine.get_current_state()}")
    
    # Start simulation
    print("\nStarting simulation...")
    engine.start_simulation()
    
    # Run dashboard (this will block)
    print("Starting dashboard on http://127.0.0.1:8051")
    print("Features:")
    print("- Real-time parameter adjustment")
    print("- Live metric updates")
    print("- Interactive scenario modeling")
    print("- WebSocket-based real-time updates")
    print("- Event logging and monitoring")
    print("- Predefined healthcare scenarios")
    
    try:
        engine.run_dashboard(port=8051, debug=False)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        engine.stop_simulation()
        print("Simulation stopped.")