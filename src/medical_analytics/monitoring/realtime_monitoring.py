"""
Real-time Monitoring and Alert System for Healthcare Metrics
Provides continuous monitoring, threshold-based alerts, and real-time data streaming
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html, dcc

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class MetricType(Enum):
    """Types of healthcare metrics to monitor"""
    PATIENT_VOLUME = "patient_volume"
    WAIT_TIMES = "wait_times"
    BED_OCCUPANCY = "bed_occupancy"
    STAFF_UTILIZATION = "staff_utilization"
    COST_PER_PATIENT = "cost_per_patient"
    READMISSION_RATE = "readmission_rate"
    MORTALITY_RATE = "mortality_rate"
    INFECTION_RATE = "infection_rate"

@dataclass
class Alert:
    """Healthcare alert data structure"""
    id: str
    metric_type: MetricType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime
    facility_id: Optional[str] = None
    resolved: bool = False

@dataclass
class MonitoringThreshold:
    """Monitoring threshold configuration"""
    metric_type: MetricType
    min_threshold: Optional[float] = None
    max_threshold: Optional[float] = None
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True

class RealTimeMonitor:
    """Real-time healthcare metrics monitoring system"""
    
    def __init__(self):
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.monitoring_active = False
        self.data_buffer = {}
        self.thresholds = self._initialize_default_thresholds()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_default_thresholds(self) -> Dict[MetricType, MonitoringThreshold]:
        """Initialize default monitoring thresholds"""
        return {
            MetricType.PATIENT_VOLUME: MonitoringThreshold(
                MetricType.PATIENT_VOLUME, max_threshold=150, severity=AlertSeverity.HIGH
            ),
            MetricType.WAIT_TIMES: MonitoringThreshold(
                MetricType.WAIT_TIMES, max_threshold=120, severity=AlertSeverity.MEDIUM
            ),
            MetricType.BED_OCCUPANCY: MonitoringThreshold(
                MetricType.BED_OCCUPANCY, max_threshold=95, severity=AlertSeverity.HIGH
            ),
            MetricType.STAFF_UTILIZATION: MonitoringThreshold(
                MetricType.STAFF_UTILIZATION, max_threshold=90, min_threshold=60, severity=AlertSeverity.MEDIUM
            ),
            MetricType.COST_PER_PATIENT: MonitoringThreshold(
                MetricType.COST_PER_PATIENT, max_threshold=15000, severity=AlertSeverity.MEDIUM
            ),
            MetricType.READMISSION_RATE: MonitoringThreshold(
                MetricType.READMISSION_RATE, max_threshold=15, severity=AlertSeverity.HIGH
            ),
            MetricType.MORTALITY_RATE: MonitoringThreshold(
                MetricType.MORTALITY_RATE, max_threshold=3, severity=AlertSeverity.CRITICAL
            ),
            MetricType.INFECTION_RATE: MonitoringThreshold(
                MetricType.INFECTION_RATE, max_threshold=5, severity=AlertSeverity.HIGH
            )
        }
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.executor.submit(self._monitoring_loop)
            logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Generate simulated real-time data
                current_data = self._generate_realtime_data()
                
                # Check thresholds and generate alerts
                self._check_thresholds(current_data)
                
                # Update data buffer
                self._update_data_buffer(current_data)
                
                # Sleep for monitoring interval
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _generate_realtime_data(self) -> Dict[str, Any]:
        """Generate simulated real-time healthcare data"""
        timestamp = datetime.now()
        
        # Simulate realistic healthcare metrics with some variability
        base_hour = timestamp.hour
        day_factor = 1.0 + 0.3 * np.sin(2 * np.pi * base_hour / 24)  # Daily pattern
        
        return {
            'timestamp': timestamp,
            'patient_volume': max(0, int(80 * day_factor + np.random.normal(0, 15))),
            'wait_times': max(0, 45 * day_factor + np.random.normal(0, 20)),
            'bed_occupancy': min(100, max(0, 75 + np.random.normal(0, 10))),
            'staff_utilization': min(100, max(0, 70 + np.random.normal(0, 8))),
            'cost_per_patient': max(0, 8000 + np.random.normal(0, 2000)),
            'readmission_rate': max(0, 8 + np.random.normal(0, 3)),
            'mortality_rate': max(0, 1.5 + np.random.normal(0, 0.5)),
            'infection_rate': max(0, 2 + np.random.normal(0, 1))
        }
    
    def _check_thresholds(self, data: Dict[str, Any]):
        """Check data against thresholds and generate alerts"""
        for metric_type, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
                
            metric_key = metric_type.value
            if metric_key not in data:
                continue
                
            value = data[metric_key]
            alert_triggered = False
            
            # Check maximum threshold
            if threshold.max_threshold and value > threshold.max_threshold:
                alert_triggered = True
                message = f"{metric_type.value.replace('_', ' ').title()} exceeded maximum threshold: {value:.2f} > {threshold.max_threshold}"
            
            # Check minimum threshold
            elif threshold.min_threshold and value < threshold.min_threshold:
                alert_triggered = True
                message = f"{metric_type.value.replace('_', ' ').title()} below minimum threshold: {value:.2f} < {threshold.min_threshold}"
            
            if alert_triggered:
                alert = Alert(
                    id=f"{metric_type.value}_{int(time.time())}",
                    metric_type=metric_type,
                    severity=threshold.severity,
                    message=message,
                    value=value,
                    threshold=threshold.max_threshold or threshold.min_threshold,
                    timestamp=data['timestamp']
                )
                
                self.active_alerts.append(alert)
                self.alert_history.append(alert)
                logger.warning(f"Alert generated: {message}")
    
    def _update_data_buffer(self, data: Dict[str, Any]):
        """Update the data buffer with new data"""
        timestamp = data['timestamp']
        
        # Initialize buffer if empty
        if not self.data_buffer:
            for key in data.keys():
                if key != 'timestamp':
                    self.data_buffer[key] = {'timestamps': [], 'values': []}
        
        # Add new data points
        for key, value in data.items():
            if key != 'timestamp':
                self.data_buffer[key]['timestamps'].append(timestamp)
                self.data_buffer[key]['values'].append(value)
                
                # Keep only last 100 data points
                if len(self.data_buffer[key]['values']) > 100:
                    self.data_buffer[key]['timestamps'] = self.data_buffer[key]['timestamps'][-100:]
                    self.data_buffer[key]['values'] = self.data_buffer[key]['values'][-100:]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts"""
        return [alert for alert in self.active_alerts if not alert.resolved]
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_realtime_metrics(self) -> Dict[str, Any]:
        """Get current real-time metrics"""
        if not self.data_buffer:
            return {}
        
        current_metrics = {}
        for metric, data in self.data_buffer.items():
            if data['values']:
                current_metrics[metric] = {
                    'current': data['values'][-1],
                    'trend': self._calculate_trend(data['values'][-10:]) if len(data['values']) >= 10 else 0,
                    'avg_1h': np.mean(data['values'][-12:]) if len(data['values']) >= 12 else data['values'][-1]
                }
        
        return current_metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        if len(values) < 2:
            return 0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return np.clip(slope / (np.std(values) + 1e-6), -1, 1)
    
    def create_monitoring_dashboard(self) -> List:
        """Create real-time monitoring dashboard layout"""
        active_alerts = self.get_active_alerts()
        metrics = self.get_realtime_metrics()
        
        return [
            # Header
            dbc.Row([
                dbc.Col([
                    html.H2("🚨 Real-time Healthcare Monitoring", className="text-primary mb-3"),
                    html.P("Live monitoring with intelligent alerts and threshold management", className="text-muted")
                ])
            ], className="mb-4"),
            
            # Alert Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔔 Active Alerts"),
                        dbc.CardBody([
                            html.Div(id="alert-summary"),
                            html.Hr(),
                            html.Div(id="alert-list")
                        ])
                    ])
                ], width=4),
                
                # Real-time Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Live Metrics"),
                        dbc.CardBody([
                            html.Div(id="realtime-metrics-summary")
                        ])
                    ])
                ], width=8)
            ], className="mb-4"),
            
            # Real-time Charts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Real-time Trends"),
                        dbc.CardBody([
                            dcc.Graph(id="realtime-trends-chart")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Threshold Configuration
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("⚙️ Threshold Configuration"),
                        dbc.CardBody([
                            html.Div(id="threshold-config")
                        ])
                    ])
                ], width=12)
            ])
        ]
    
    def create_realtime_trends_chart(self) -> go.Figure:
        """Create real-time trends chart"""
        if not self.data_buffer:
            return go.Figure().add_annotation(text="No real-time data available", showarrow=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Patient Volume', 'Wait Times (min)', 'Bed Occupancy (%)', 'Staff Utilization (%)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics_to_plot = [
            ('patient_volume', 1, 1),
            ('wait_times', 1, 2),
            ('bed_occupancy', 2, 1),
            ('staff_utilization', 2, 2)
        ]
        
        for metric, row, col in metrics_to_plot:
            if metric in self.data_buffer:
                data = self.data_buffer[metric]
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamps'],
                        y=data['values'],
                        mode='lines+markers',
                        name=metric.replace('_', ' ').title(),
                        line=dict(width=2)
                    ),
                    row=row, col=col
                )
                
                # Add threshold lines if available
                threshold = self.thresholds.get(MetricType(metric))
                if threshold:
                    if threshold.max_threshold:
                        fig.add_hline(
                            y=threshold.max_threshold,
                            line_dash="dash",
                            line_color="red",
                            row=row, col=col
                        )
                    if threshold.min_threshold:
                        fig.add_hline(
                            y=threshold.min_threshold,
                            line_dash="dash",
                            line_color="orange",
                            row=row, col=col
                        )
        
        fig.update_layout(
            title="Real-time Healthcare Metrics Trends",
            template='plotly_white',
            height=600,
            showlegend=False
        )
        
        return fig

class AlertManager:
    """Manages healthcare alerts and notifications"""
    
    def __init__(self):
        self.notification_channels = []
        self.alert_rules = {}
    
    def add_notification_channel(self, channel_type: str, config: Dict):
        """Add notification channel (email, SMS, webhook, etc.)"""
        self.notification_channels.append({
            'type': channel_type,
            'config': config,
            'enabled': True
        })
    
    def send_alert(self, alert: Alert):
        """Send alert through configured channels"""
        for channel in self.notification_channels:
            if channel['enabled']:
                try:
                    self._send_via_channel(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel['type']}: {e}")
    
    def _send_via_channel(self, alert: Alert, channel: Dict):
        """Send alert via specific channel"""
        # Implementation would depend on channel type
        # For now, just log the alert
        logger.info(f"Alert sent via {channel['type']}: {alert.message}")

# Global monitor instance
healthcare_monitor = RealTimeMonitor()
alert_manager = AlertManager()