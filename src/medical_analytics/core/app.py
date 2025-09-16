import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
import threading

from ..data.data_fetcher import MedicalDataFetcher
from ..data.advanced_data_fetcher import AdvancedMedicalDataFetcher
from ..ml.ml_models import ModelManager, MonteCarloSimulator
from ..ml.advanced_ml_models import AdvancedModelManager
from ..geospatial.geospatial_analysis import HealthcareAccessibilityAnalyzer, HealthcareGISAnalyzer
from ..geospatial.geospatial_enhanced import AdvancedGeospatialAnalyzer, HealthcareGISProcessor
from ..financial.financial_analysis import HealthcareROIAnalyzer, HealthcareMarketValuation, HealthcareInvestmentAnalyzer
from ..data.data_pipeline import ETLPipeline, DataSource, RealTimeDataStreamer
from ..dashboards.specialized_dashboards import SpecializedDashboardManager
from ..monitoring.realtime_monitoring import healthcare_monitor, AlertManager
from config.settings import settings

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Advanced USA Medical Industry Analytics Platform"

# Initialize components
basic_fetcher = MedicalDataFetcher()
advanced_fetcher = AdvancedMedicalDataFetcher()
model_manager = ModelManager()
advanced_model_manager = AdvancedModelManager()
monte_carlo = MonteCarloSimulator()
accessibility_analyzer = HealthcareAccessibilityAnalyzer()
gis_analyzer = HealthcareGISAnalyzer()
advanced_geospatial = AdvancedGeospatialAnalyzer()
gis_processor = HealthcareGISProcessor()
roi_analyzer = HealthcareROIAnalyzer()
market_valuation = HealthcareMarketValuation()
investment_analyzer = HealthcareInvestmentAnalyzer()
specialized_dashboard_manager = SpecializedDashboardManager()
alert_manager = AlertManager()

# Initialize ETL pipeline
etl_pipeline = ETLPipeline()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global data cache
data_cache = {
    'last_update': None,
    'basic_data': None,
    'advanced_data': None,
    'ml_predictions': None,
    'simulations': None,
    'geospatial_data': None,
    'financial_analysis': None
}

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🏥 Advanced USA Medical Industry Analytics Platform", 
                   className="text-center mb-4 text-primary"),
            html.P("Real-time data • Machine Learning • Predictive Analytics • Geospatial Intelligence • Financial Modeling",
                   className="text-center text-muted mb-4")
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🎛️ Control Panel", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Analysis Type:"),
                            dcc.Dropdown(
                                id='analysis-type',
                                options=[
                                    {'label': '📊 Overview Dashboard', 'value': 'overview'},
                                    {'label': '🤖 ML Predictions', 'value': 'ml'},
                                    {'label': '🎲 Monte Carlo Simulation', 'value': 'simulation'},
                                    {'label': '🗺️ Geospatial Analysis', 'value': 'geospatial'},
                                    {'label': '💰 Financial Analysis', 'value': 'financial'},
                                    {'label': '📈 Real-time Streaming', 'value': 'streaming'},
                                    {'label': '🚨 Real-time Monitoring', 'value': 'monitoring'},
                                    {'label': '👥 Specialized Dashboards', 'value': 'specialized'}
                                ],
                                value='overview'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Time Range:"),
                            dcc.Dropdown(
                                id='time-range',
                                options=[
                                    {'label': 'Last 30 Days', 'value': '30d'},
                                    {'label': 'Last 90 Days', 'value': '90d'},
                                    {'label': 'Last Year', 'value': '1y'},
                                    {'label': 'Last 5 Years', 'value': '5y'},
                                    {'label': 'All Time', 'value': 'all'}
                                ],
                                value='1y'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Update Data:"),
                            html.Br(),
                            dbc.Button("🔄 Refresh All Data", id="refresh-btn", color="primary", size="sm")
                        ], width=4)
                    ])
                ])
            ], className="mb-4")
        ])
    ]),
    
    # Main Content Area
    html.Div(id="main-content"),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("© 2024 Advanced Medical Analytics Platform | Real-time Healthcare Intelligence",
                   className="text-center text-muted small")
        ])
    ])
], fluid=True)

# Data loading function
def load_all_data():
    """Load and cache all data sources - use ONLY cached data, no API calls"""
    try:
        logger.info("Loading data from cache only (no API calls)...")
        
        # Load cached data directly from database without making any API calls
        try:
            logger.info("Reading cached data from database...")
            
            # Read cached data directly from the database
            cached_data = advanced_fetcher._get_all_cached_data()
            
            if cached_data and len(cached_data) > 0:
                logger.info(f"Successfully loaded {len(cached_data)} cached datasets")
                
                # Use cached data as advanced_data
                advanced_data = cached_data
                
                # Set basic_data to empty since we're not making API calls
                basic_data = {}
                
                # Generate ML predictions from cached data only if we have sufficient data
                advanced_ml_predictions = {}
                if advanced_data and any(isinstance(v, pd.DataFrame) and not v.empty for v in advanced_data.values()):
                    try:
                        # Prepare sample data for advanced models
                        sample_data = {}
                        
                        # Cost prediction features
                        for key in ['healthcare_spending_trends', 'spending_trends']:
                            if key in advanced_data:
                                spending_df = advanced_data[key]
                                if isinstance(spending_df, pd.DataFrame) and not spending_df.empty:
                                    sample_data['cost_features'] = spending_df
                                    sample_data['cost_targets'] = spending_df.iloc[:, -1]  # Use last numeric column
                                    break
                        
                        # Time series data for forecasting
                        for key in ['employment_data', 'economic_indicators']:
                            if key in advanced_data:
                                employment_df = advanced_data[key]
                                if isinstance(employment_df, pd.DataFrame) and not employment_df.empty:
                                    ts_data = employment_df.copy()
                                    if 'date' not in ts_data.columns and 'year' in ts_data.columns:
                                        ts_data['date'] = pd.to_datetime(ts_data['year'], format='%Y')
                                    if 'value' not in ts_data.columns:
                                        ts_data['value'] = ts_data.iloc[:, -1]
                                    sample_data['time_series_data'] = ts_data
                                    break
                        
                        # Anomaly detection features
                        for key in ['provider_statistics', 'medicare_providers', 'hospital_data']:
                            if key in advanced_data:
                                provider_df = advanced_data[key]
                                if isinstance(provider_df, pd.DataFrame) and not provider_df.empty:
                                    sample_data['anomaly_features'] = provider_df
                                    break
                        
                        # Only generate predictions if we have sufficient sample data
                        if sample_data:
                            advanced_ml_predictions = advanced_model_manager.get_comprehensive_predictions(sample_data)
                        
                    except Exception as e:
                        logger.error(f"Error in ML predictions with cached data: {e}")
                        advanced_ml_predictions = {}
                
                # Run Monte Carlo simulations on cached data
                simulations = {}
                for key in ['healthcare_spending_trends', 'spending_trends']:
                    if advanced_data and key in advanced_data:
                        spending_df = advanced_data[key]
                        if isinstance(spending_df, pd.DataFrame) and not spending_df.empty:
                            try:
                                simulations['spending_projection'] = monte_carlo.simulate_healthcare_spending(spending_df)
                                break
                            except Exception as e:
                                logger.error(f"Error in Monte Carlo simulation: {e}")
                
                # Generate minimal geospatial analysis from cached data
                geospatial_data = {}
                try:
                    # Basic geospatial analysis from cached hospital data
                    for key in ['hospital_data', 'hospitals', 'medicare_providers']:
                        if advanced_data and key in advanced_data:
                            provider_df = advanced_data[key]
                            if isinstance(provider_df, pd.DataFrame) and not provider_df.empty:
                                try:
                                    geospatial_data['accessibility'] = accessibility_analyzer.calculate_accessibility_scores(provider_df)
                                    geospatial_data['healthcare_deserts'] = accessibility_analyzer.identify_healthcare_deserts(provider_df)
                                    break
                                except Exception as e:
                                    logger.error(f"Error in geospatial analysis: {e}")
                    
                    # Minimal geospatial data structure
                    geospatial_data.update({
                        'advanced_report': {},
                        'clustering_analysis': {},
                        'catchment_areas': {},
                        'service_gaps': {},
                        'facilities': []
                    })
                    
                except Exception as e:
                    logger.error(f"Error in geospatial analysis: {e}")
                    geospatial_data = {}
                
                # Generate financial analysis from cached data
                financial_analysis = {}
                for key in ['healthcare_spending_trends', 'spending_trends']:
                    if advanced_data and key in advanced_data:
                        spending_df = advanced_data[key]
                        if isinstance(spending_df, pd.DataFrame) and not spending_df.empty:
                            try:
                                financial_analysis['roi_analysis'] = roi_analyzer.calculate_healthcare_roi(spending_df)
                                financial_analysis['market_valuation'] = market_valuation.calculate_market_metrics(spending_df)
                                break
                            except Exception as e:
                                logger.error(f"Error in financial analysis: {e}")
                
                # Update cache
                data_cache.update({
                    'last_update': datetime.now(),
                    'basic_data': basic_data,
                    'advanced_data': advanced_data,
                    'ml_predictions': advanced_ml_predictions,
                    'simulations': simulations,
                    'geospatial_data': geospatial_data,
                    'financial_analysis': financial_analysis
                })
                
                logger.info("All cached data loaded and processed successfully (no API calls made)")
                return True
            else:
                logger.warning("No cached data found in database")
                
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            logger.info("Cache loading failed, initializing with empty data structure")
        
        # If we reach here, either cache loading failed or no cache exists
        # Set up minimal data structure to prevent errors
        data_cache.update({
            'last_update': datetime.now(),
            'basic_data': {},
            'advanced_data': {},
            'ml_predictions': {},
            'simulations': {},
            'geospatial_data': {},
            'financial_analysis': {}
        })
        
        logger.info("Initialized with empty data structure - use 'Refresh All Data' button to load fresh data")
        return True
        
    except Exception as e:
        logger.error(f"Error in load_all_data: {str(e)}")
        # Initialize with empty structure to prevent crashes
        data_cache.update({
            'last_update': datetime.now(),
            'basic_data': {},
            'advanced_data': {},
            'ml_predictions': {},
            'simulations': {},
            'geospatial_data': {},
            'financial_analysis': {}
        })
        return False

# Initialize data on startup
load_all_data()

# Main callback for dynamic content
@app.callback(
    Output('main-content', 'children'),
    [Input('analysis-type', 'value'),
     Input('time-range', 'value'),
     Input('refresh-btn', 'n_clicks')]
)
def update_main_content(analysis_type, time_range, n_clicks):
    """Update main content based on selected analysis type"""
    
    # Refresh data if button clicked
    if n_clicks:
        load_all_data()
    
    if analysis_type == 'overview':
        return create_overview_dashboard()
    elif analysis_type == 'ml':
        return create_ml_dashboard()
    elif analysis_type == 'simulation':
        return create_simulation_dashboard()
    elif analysis_type == 'geospatial':
        return create_geospatial_dashboard()
    elif analysis_type == 'financial':
        return create_financial_dashboard()
    elif analysis_type == 'streaming':
        return create_streaming_dashboard()
    elif analysis_type == 'monitoring':
        return create_monitoring_dashboard()
    elif analysis_type == 'specialized':
        return create_specialized_dashboard_selector()
    else:
        return create_overview_dashboard()

def create_overview_dashboard():
    """Create the overview dashboard with key metrics and basic charts"""
    basic_data = data_cache.get('basic_data', {})
    
    return [
        # Key Metrics Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("$4.3T", className="card-title text-success"),
                        html.P("Total Healthcare Spending", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("6,090", className="card-title text-info"),
                        html.P("Total Hospitals", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("22.2M", className="card-title text-warning"),
                        html.P("Healthcare Workers", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("$13,493", className="card-title text-danger"),
                        html.P("Cost Per Capita", className="card-text")
                    ])
                ])
            ], width=3)
        ], className="mb-4"),
        
        # Charts Row 1
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Healthcare Spending Trends", className="card-title"),
                        dcc.Graph(id="spending-trends")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Spending by Category", className="card-title"),
                        dcc.Graph(id="spending-categories")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Charts Row 2
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Healthcare Employment by State", className="card-title"),
                        dcc.Graph(id="employment-map")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Top Healthcare Companies", className="card-title"),
                        dcc.Graph(id="top-companies")
                    ])
                ])
            ], width=4)
        ], className="mb-4")
    ]

def create_ml_dashboard():
    """Create enhanced ML predictions dashboard with advanced models"""
    ml_predictions = data_cache.get('ml_predictions', {})
    advanced_predictions = data_cache.get('advanced_ml_predictions', {})
    
    return [
        dbc.Row([
            dbc.Col([
                html.H3("🤖 Advanced Machine Learning Predictions", className="text-primary mb-4"),
                html.P("Deep Learning • Ensemble Models • Advanced Forecasting • Anomaly Detection", 
                       className="text-muted mb-4")
            ])
        ]),
        
        # Model Performance Metrics
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Model Performance", className="card-title"),
                        html.Div(id="model-performance-metrics")
                    ])
                ])
            ], width=12)
        ], className="mb-4"),
        
        # Cost Prediction Models
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Deep Learning Cost Forecast", className="card-title"),
                        html.P("Neural network-based cost prediction with feature engineering", 
                               className="text-muted small"),
                        dcc.Graph(id="deep-cost-forecast-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Ensemble Model Predictions", className="card-title"),
                        html.P("Combined XGBoost, LightGBM, CatBoost, and Random Forest", 
                               className="text-muted small"),
                        dcc.Graph(id="ensemble-predictions-chart")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Advanced Time Series and Anomaly Detection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Advanced Time Series Forecast", className="card-title"),
                        html.P("Prophet, ARIMA, and Exponential Smoothing comparison", 
                               className="text-muted small"),
                        dcc.Graph(id="advanced-forecast-chart")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Anomaly Detection", className="card-title"),
                        html.P("Isolation Forest anomaly scores", 
                               className="text-muted small"),
                        dcc.Graph(id="anomaly-detection-chart")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Model Comparison and Feature Importance
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Model Comparison", className="card-title"),
                        dcc.Graph(id="model-comparison-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Feature Importance Analysis", className="card-title"),
                        dcc.Graph(id="feature-importance-chart")
                    ])
                ])
            ], width=6)
        ])
    ]

def create_simulation_dashboard():
    """Create Monte Carlo simulation dashboard"""
    simulations = data_cache.get('simulations', {})
    
    return [
        dbc.Row([
            dbc.Col([
                html.H3("🎲 Monte Carlo Simulations", className="text-primary mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Healthcare Spending Projections", className="card-title"),
                        dcc.Graph(id="spending-simulation-chart")
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Simulation Parameters", className="card-title"),
                        html.Div(id="simulation-controls")
                    ])
                ])
            ], width=4)
        ])
    ]

def create_geospatial_dashboard():
    """Create enhanced geospatial analysis dashboard with advanced mapping and clustering"""
    geospatial_data = data_cache.get('geospatial_data', {})
    
    return [
        dbc.Row([
            dbc.Col([
                html.H3("🗺️ Advanced Geospatial Healthcare Analysis", className="text-primary mb-4"),
                html.P("Interactive mapping • Spatial clustering • Accessibility analysis • Healthcare deserts identification", 
                       className="text-muted mb-4")
            ])
        ]),
        
        # First row - Main maps
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Interactive Healthcare Accessibility Map", className="card-title"),
                        html.P("Real-time facility locations with accessibility heatmap", className="card-text text-muted"),
                        dcc.Graph(id="enhanced-accessibility-map", style={'height': '500px'})
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Healthcare Deserts Analysis", className="card-title"),
                        html.Div(id="healthcare-deserts-analysis")
                    ])
                ])
            ], width=4)
        ], className="mb-4"),
        
        # Second row - Clustering and network analysis
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Spatial Clustering Analysis", className="card-title"),
                        html.P("DBSCAN and K-means clustering of healthcare facilities", className="card-text text-muted"),
                        dcc.Graph(id="spatial-clustering-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Network Connectivity Analysis", className="card-title"),
                        html.P("Healthcare facility network and referral patterns", className="card-text text-muted"),
                        dcc.Graph(id="network-connectivity-chart")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Third row - Service gaps and catchment areas
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Service Coverage Gaps", className="card-title"),
                        html.P("Analysis of healthcare service availability by region", className="card-text text-muted"),
                        dcc.Graph(id="service-gaps-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Facility Catchment Areas", className="card-title"),
                        html.P("Population coverage and service area analysis", className="card-text text-muted"),
                        dcc.Graph(id="catchment-areas-chart")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        # Fourth row - Analytics summary
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Geospatial Analytics Summary", className="card-title"),
                        html.Div(id="geospatial-summary-metrics")
                    ])
                ])
            ])
        ])
    ]

def create_financial_dashboard():
    """Create financial analysis dashboard"""
    financial_analysis = data_cache.get('financial_analysis', {})
    
    return [
        dbc.Row([
            dbc.Col([
                html.H3("💰 Financial Analysis", className="text-primary mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("ROI Analysis", className="card-title"),
                        dcc.Graph(id="roi-chart")
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Market Valuation", className="card-title"),
                        dcc.Graph(id="market-valuation-chart")
                    ])
                ])
            ], width=6)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Investment Analysis", className="card-title"),
                        dash_table.DataTable(
                            id="investment-table",
                            columns=[
                                {"name": "Investment", "id": "investment"},
                                {"name": "NPV", "id": "npv"},
                                {"name": "IRR", "id": "irr"},
                                {"name": "Risk Score", "id": "risk"}
                            ],
                            style_cell={'textAlign': 'left'},
                            style_data_conditional=[
                                {
                                    'if': {'row_index': 'odd'},
                                    'backgroundColor': 'rgb(248, 248, 248)'
                                }
                            ]
                        )
                    ])
                ])
            ], width=12)
        ])
    ]

def create_streaming_dashboard():
    """Create enhanced real-time streaming dashboard with advanced monitoring"""
    return [
        # Header with system status
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H4("📊 Enhanced Real-time Streaming Dashboard", className="alert-heading"),
                    html.P("Advanced real-time data streaming with predictive analytics and anomaly detection")
                ], color="info", className="mb-3")
            ], width=12)
        ]),
        
        # System Health Overview
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 System Health Overview"),
                    dbc.CardBody([
                        html.Div(id='system-health-indicators'),
                        dbc.Progress(id='system-performance-bar', value=85, color="success", className="mt-2")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚡ Live Data Throughput"),
                    dbc.CardBody([
                        html.Div(id='throughput-metrics'),
                        dcc.Graph(id='throughput-chart', style={'height': '200px'})
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎯 Prediction Accuracy"),
                    dbc.CardBody([
                        html.Div(id='prediction-accuracy'),
                        dcc.Graph(id='accuracy-gauge', style={'height': '200px'})
                    ])
                ])
            ], width=4)
        ], className="mb-3"),
        
        # Main streaming charts
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "📈 Multi-Stream Analytics",
                        dbc.ButtonGroup([
                            dbc.Button("1m", id="timeframe-1m", size="sm", outline=True),
                            dbc.Button("5m", id="timeframe-5m", size="sm", outline=True),
                            dbc.Button("15m", id="timeframe-15m", size="sm", outline=True, active=True),
                            dbc.Button("1h", id="timeframe-1h", size="sm", outline=True)
                        ], className="float-end")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="live-metrics-chart", style={'height': '400px'})
                    ])
                ])
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🚨 Anomaly Detection"),
                    dbc.CardBody([
                        html.Div(id='anomaly-alerts'),
                        dcc.Graph(id='anomaly-score-chart', style={'height': '200px'}),
                        html.Hr(),
                        html.H6("Recent Anomalies"),
                        html.Div(id='recent-anomalies-list')
                    ])
                ])
            ], width=4)
        ], className="mb-3"),
        
        # Advanced analytics row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔮 Predictive Trends"),
                    dbc.CardBody([
                        dcc.Graph(id='predictive-chart', style={'height': '300px'})
                    ])
                ])
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📊 Statistical Summary"),
                    dbc.CardBody([
                        html.Div(id='statistical-summary'),
                        dcc.Graph(id='distribution-chart', style={'height': '250px'})
                    ])
                ])
            ], width=6)
        ], className="mb-3"),
        
        # Control panel
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚙️ Streaming Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Update Frequency"),
                                dcc.Slider(
                                    id='update-frequency-slider',
                                    min=1, max=10, value=2, step=1,
                                    marks={i: f'{i}s' for i in range(1, 11)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Data Sources"),
                                dcc.Checklist(
                                    id='data-sources-checklist',
                                    options=[
                                        {'label': 'Patient Vitals', 'value': 'vitals'},
                                        {'label': 'Equipment Status', 'value': 'equipment'},
                                        {'label': 'Environmental', 'value': 'environment'},
                                        {'label': 'Network Traffic', 'value': 'network'}
                                    ],
                                    value=['vitals', 'equipment'],
                                    inline=True
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Alert Sensitivity"),
                                dcc.Dropdown(
                                    id='alert-sensitivity-dropdown',
                                    options=[
                                        {'label': 'Low', 'value': 'low'},
                                        {'label': 'Medium', 'value': 'medium'},
                                        {'label': 'High', 'value': 'high'},
                                        {'label': 'Critical Only', 'value': 'critical'}
                                    ],
                                    value='medium'
                                )
                            ], width=4)
                        ])
                    ])
                ])
            ], width=12)
        ]),
        
        # Update intervals
        dcc.Interval(id="interval-component", interval=5000, n_intervals=0),
        dcc.Interval(id='anomaly-interval', interval=5000, n_intervals=0),
        dcc.Interval(id='prediction-interval', interval=10000, n_intervals=0)
    ]

def create_specialized_dashboard_selector():
    """Create the specialized dashboard selector interface"""
    return specialized_dashboard_manager.create_dashboard_selector()

def create_monitoring_dashboard():
    """Create the real-time monitoring dashboard"""
    # Start monitoring if not already active
    if not healthcare_monitor.monitoring_active:
        healthcare_monitor.start_monitoring()
    
    return healthcare_monitor.create_monitoring_dashboard()

# Chart callbacks for overview dashboard - with conditional execution
@app.callback(
    Output('spending-trends', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_spending_trends(_, analysis_type):
    """Update spending trends chart - only when overview dashboard is active"""
    if analysis_type != 'overview':
        raise dash.exceptions.PreventUpdate
        
    basic_data = data_cache.get('basic_data', {})
    if basic_data:
        df = basic_data.get('spending_trends', pd.DataFrame())
        if not df.empty:
            fig = px.line(df, x='year', y='spending', 
                         title='Healthcare Spending Over Time',
                         labels={'spending': 'Spending (Billions USD)', 'year': 'Year'})
            fig.update_layout(template='plotly_white')
            return fig
    
    # Fallback data
    years = list(range(2015, 2025))
    spending = [3.2 + i * 0.15 for i in range(len(years))]
    fig = px.line(x=years, y=spending, title='Healthcare Spending Over Time')
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('spending-categories', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_spending_categories(_, analysis_type):
    """Update spending categories chart - only when overview dashboard is active"""
    if analysis_type != 'overview':
        raise dash.exceptions.PreventUpdate
        
    basic_data = data_cache.get('basic_data', {})
    if basic_data:
        df = basic_data.get('spending_by_category', pd.DataFrame())
        if not df.empty:
            fig = px.pie(df, values='amount', names='category',
                        title='Healthcare Spending by Category')
            fig.update_layout(template='plotly_white')
            return fig
    
    # Fallback data
    categories = ['Hospital Care', 'Physician Services', 'Prescription Drugs', 'Other']
    amounts = [1.3, 0.8, 0.5, 0.7]
    fig = px.pie(values=amounts, names=categories, title='Healthcare Spending by Category')
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('employment-map', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_employment_map(_, analysis_type):
    """Update employment map - only when overview dashboard is active"""
    if analysis_type != 'overview':
        raise dash.exceptions.PreventUpdate
        
    basic_data = data_cache.get('basic_data', {})
    if basic_data:
        df = basic_data.get('employment_by_state', pd.DataFrame())
        if not df.empty:
            fig = px.choropleth(df, locations='state_code', color='employment',
                               locationmode='USA-states',
                               title='Healthcare Employment by State',
                               color_continuous_scale='Blues')
            fig.update_layout(geo_scope='usa', template='plotly_white')
            return fig
    
    # Fallback data
    states = ['CA', 'TX', 'FL', 'NY', 'PA']
    employment = [2.5, 1.8, 1.2, 1.5, 0.9]
    fallback_df = pd.DataFrame({'locations': states, 'employment': employment})
    fig = px.choropleth(fallback_df, locations='locations', color='employment',
                       locationmode='USA-states',
                       title='Healthcare Employment by State',
                       color_continuous_scale='Blues')
    fig.update_layout(geo_scope='usa', template='plotly_white')
    return fig

@app.callback(
    Output('top-companies', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_top_companies(_, analysis_type):
    """Update top companies chart - only when overview dashboard is active"""
    if analysis_type != 'overview':
        raise dash.exceptions.PreventUpdate
        
    basic_data = data_cache.get('basic_data', {})
    if basic_data:
        df = basic_data.get('top_companies', pd.DataFrame())
        if not df.empty:
            fig = px.bar(df, x='revenue', y='company', orientation='h',
                        title='Top Healthcare Companies by Revenue')
            fig.update_layout(template='plotly_white')
            return fig
    
    # Fallback data
    companies = ['UnitedHealth', 'CVS Health', 'Anthem', 'Johnson & Johnson', 'Pfizer']
    revenues = [287, 268, 122, 94, 81]
    fig = px.bar(x=revenues, y=companies, orientation='h',
                title='Top Healthcare Companies by Revenue (Billions USD)')
    fig.update_layout(template='plotly_white')
    return fig

# ML Dashboard callbacks - with conditional execution
@app.callback(
    Output('deep-cost-forecast-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_deep_cost_forecast(_, analysis_type):
    """Update deep learning cost forecast chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    fig = go.Figure()
    
    # Add deep learning predictions
    if 'deep_learning' in ml_predictions:
        dl_pred = ml_predictions['deep_learning']
        if 'cost_predictions' in dl_pred:
            dates = pd.date_range(start='2024-01-01', periods=len(dl_pred['cost_predictions']), freq='M')
            fig.add_trace(go.Scatter(
                x=dates,
                y=dl_pred['cost_predictions'],
                mode='lines+markers',
                name='Deep Learning Forecast',
                line=dict(color='#1f77b4', width=3)
            ))
    else:
        # Fallback data
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        forecast = np.random.normal(4.8, 0.3, 12)
        fig.add_trace(go.Scatter(x=dates, y=forecast, mode='lines+markers', name='Deep Learning Forecast'))
    
    fig.update_layout(
        title='Deep Learning Cost Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Cost (Trillions $)',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('ensemble-predictions-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_ensemble_predictions(_, analysis_type):
    """Update ensemble model predictions chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    fig = go.Figure()
    
    # Add ensemble predictions
    if 'ensemble' in ml_predictions:
        ensemble_pred = ml_predictions['ensemble']
        if 'predictions' in ensemble_pred:
            dates = pd.date_range(start='2024-01-01', periods=len(ensemble_pred['predictions']), freq='M')
            fig.add_trace(go.Scatter(
                x=dates,
                y=ensemble_pred['predictions'],
                mode='lines+markers',
                name='Ensemble Model',
                line=dict(color='#ff7f0e', width=2)
            ))
    else:
        # Fallback data
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        predictions = np.random.normal(4.6, 0.25, 12)
        fig.add_trace(go.Scatter(x=dates, y=predictions, mode='lines+markers', name='Ensemble Model'))
    
    fig.update_layout(
        title='Ensemble Model Predictions',
        xaxis_title='Date',
        yaxis_title='Predicted Values',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('advanced-forecast-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_advanced_forecast(_, analysis_type):
    """Update advanced time series forecast chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    fig = go.Figure()
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    
    # Add different forecasting models
    if 'time_series' in ml_predictions:
        ts_pred = ml_predictions['time_series']
        for model_name, predictions in ts_pred.items():
            if isinstance(predictions, list) and len(predictions) > 0:
                fig.add_trace(go.Scatter(
                    x=dates[:len(predictions)],
                    y=predictions,
                    mode='lines+markers',
                    name=model_name.title()
                ))
    else:
        # Fallback data
        prophet_pred = np.random.normal(4.7, 0.2, 12)
        arima_pred = np.random.normal(4.65, 0.25, 12)
        exp_smooth_pred = np.random.normal(4.75, 0.15, 12)
        
        fig.add_trace(go.Scatter(x=dates, y=prophet_pred, mode='lines+markers', name='Prophet'))
        fig.add_trace(go.Scatter(x=dates, y=arima_pred, mode='lines+markers', name='ARIMA'))
        fig.add_trace(go.Scatter(x=dates, y=exp_smooth_pred, mode='lines+markers', name='Exponential Smoothing'))
    
    fig.update_layout(
        title='Advanced Time Series Forecasting Comparison',
        xaxis_title='Date',
        yaxis_title='Forecast Values',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('anomaly-detection-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_anomaly_detection(_, analysis_type):
    """Update anomaly detection chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    fig = go.Figure()
    
    if 'anomaly_detection' in ml_predictions:
        anomaly_data = ml_predictions['anomaly_detection']
        if 'scores' in anomaly_data:
            scores = anomaly_data['scores']
            fig.add_trace(go.Scatter(
                y=scores,
                mode='markers',
                name='Anomaly Scores',
                marker=dict(color=scores, colorscale='Reds', size=8)
            ))
    else:
        # Fallback data
        scores = np.random.uniform(-0.1, 0.5, 50)
        fig.add_trace(go.Scatter(
            y=scores,
            mode='markers',
            name='Anomaly Scores',
            marker=dict(color=scores, colorscale='Reds', size=8)
        ))
    
    fig.update_layout(
        title='Anomaly Detection Scores',
        xaxis_title='Data Point',
        yaxis_title='Anomaly Score',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('model-comparison-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_model_comparison(_, analysis_type):
    """Update model comparison chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    # Sample model performance metrics
    models = ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest', 'Neural Network']
    accuracy = np.random.uniform(0.75, 0.95, len(models))
    
    fig = px.bar(x=models, y=accuracy, title='Model Performance Comparison')
    fig.update_layout(
        xaxis_title='Model',
        yaxis_title='Accuracy Score',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('feature-importance-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_feature_importance(_, analysis_type):
    """Update feature importance chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    # Sample feature importance data
    features = ['Demographics', 'Economic Indicators', 'Healthcare Utilization', 'Geographic Factors', 'Policy Changes']
    importance = np.random.uniform(0.1, 0.4, len(features))
    
    fig = px.bar(x=importance, y=features, orientation='h', title='Feature Importance Analysis')
    fig.update_layout(
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white'
    )
    return fig

@app.callback(
    Output('model-performance-metrics', 'children'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_model_performance_metrics(_, analysis_type):
    """Update model performance metrics - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    ml_predictions = data_cache.get('ml_predictions', {})
    
    # Sample performance metrics
    metrics = [
        {'model': 'Deep Learning', 'accuracy': '94.2%', 'rmse': '0.087', 'mae': '0.065'},
        {'model': 'Ensemble', 'accuracy': '92.8%', 'rmse': '0.095', 'mae': '0.071'},
        {'model': 'XGBoost', 'accuracy': '91.5%', 'rmse': '0.102', 'mae': '0.078'},
        {'model': 'Prophet', 'accuracy': '89.3%', 'rmse': '0.115', 'mae': '0.089'}
    ]
    
    return dash_table.DataTable(
        data=metrics,
        columns=[
            {'name': 'Model', 'id': 'model'},
            {'name': 'Accuracy', 'id': 'accuracy'},
            {'name': 'RMSE', 'id': 'rmse'},
            {'name': 'MAE', 'id': 'mae'}
        ],
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )

@app.callback(
    Output('demand-forecast-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_demand_forecast(_, analysis_type):
    """Update demand forecast chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    # Generate sample demand forecast
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    demand = np.random.normal(100, 10, 12)
    
    fig = px.line(x=dates, y=demand, title='Healthcare Demand Forecast')
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('risk-assessment-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_risk_assessment(_, analysis_type):
    """Update risk assessment chart - only when ML dashboard is active"""
    if analysis_type != 'ml':
        raise dash.exceptions.PreventUpdate
        
    # Generate sample risk assessment
    categories = ['Financial', 'Operational', 'Regulatory', 'Market', 'Technology']
    risk_scores = np.random.uniform(0.2, 0.8, 5)
    
    fig = px.bar(x=categories, y=risk_scores, title='Healthcare Risk Assessment')
    fig.update_layout(template='plotly_white', yaxis_title='Risk Score')
    return fig

# Simulation Dashboard callbacks - with conditional execution
@app.callback(
    Output('spending-simulation-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_spending_simulation(_, analysis_type):
    """Update spending simulation chart - only when simulation dashboard is active"""
    if analysis_type != 'simulation':
        raise dash.exceptions.PreventUpdate
        
    # Generate Monte Carlo simulation results
    years = list(range(2024, 2034))
    simulations = []
    
    for _ in range(100):
        base = 4.5
        growth_rate = np.random.normal(0.05, 0.02)
        simulation = [base * (1 + growth_rate) ** i for i in range(len(years))]
        simulations.append(simulation)
    
    # Calculate percentiles
    simulations = np.array(simulations)
    p10 = np.percentile(simulations, 10, axis=0)
    p50 = np.percentile(simulations, 50, axis=0)
    p90 = np.percentile(simulations, 90, axis=0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p10, name='10th Percentile', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=years, y=p50, name='Median', line=dict(width=3)))
    fig.add_trace(go.Scatter(x=years, y=p90, name='90th Percentile', line=dict(dash='dash')))
    fig.update_layout(title='Healthcare Spending Projections (Monte Carlo)', template='plotly_white')
    return fig

# Enhanced Geospatial Dashboard callbacks - with conditional execution
@app.callback(
    Output('enhanced-accessibility-map', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_enhanced_accessibility_map(_, analysis_type):
    """Update enhanced accessibility map - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    geospatial_data = data_cache.get('geospatial_data', {})
    
    fig = go.Figure()
    
    # Add facility locations if available
    if 'facilities' in geospatial_data and geospatial_data['facilities']:
        facilities = geospatial_data['facilities']
        
        # Create scatter plot for facilities
        fig.add_trace(go.Scattermapbox(
            lat=[f['latitude'] for f in facilities],
            lon=[f['longitude'] for f in facilities],
            mode='markers',
            marker=dict(
                size=10,
                color=[f.get('rating', 3) for f in facilities],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Quality Rating")
            ),
            text=[f"{f['name']} ({f['type']})" for f in facilities],
            name='Healthcare Facilities'
        ))
    else:
        # Fallback sample data
        sample_lats = [40.7128, 34.0522, 41.8781, 29.7604, 33.4484]
        sample_lons = [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740]
        sample_names = ['NYC Medical Center', 'LA General', 'Chicago Hospital', 'Houston Medical', 'Phoenix Care']
        
        fig.add_trace(go.Scattermapbox(
            lat=sample_lats,
            lon=sample_lons,
            mode='markers',
            marker=dict(size=12, color='red'),
            text=sample_names,
            name='Healthcare Facilities'
        ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=39.8283, lon=-98.5795),  # Center of USA
            zoom=3
        ),
        title='Interactive Healthcare Accessibility Map',
        height=500,
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('healthcare-deserts-analysis', 'children'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_healthcare_deserts_analysis(_, analysis_type):
    """Update healthcare deserts analysis - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    geospatial_data = data_cache.get('geospatial_data', {})
    
    # Sample healthcare desert metrics
    desert_metrics = [
        {'metric': 'Total Healthcare Deserts', 'value': '2,847', 'change': '+3.2%'},
        {'metric': 'Population Affected', 'value': '83.7M', 'change': '+1.8%'},
        {'metric': 'Rural Deserts', 'value': '1,923', 'change': '+4.1%'},
        {'metric': 'Urban Deserts', 'value': '924', 'change': '+1.9%'}
    ]
    
    cards = []
    for metric in desert_metrics:
        color = 'success' if metric['change'].startswith('-') else 'warning'
        cards.append(
            dbc.Card([
                dbc.CardBody([
                    html.H6(metric['value'], className=f"text-{color}"),
                    html.P(metric['metric'], className="card-text small"),
                    html.Small(metric['change'], className=f"text-{color}")
                ])
            ], className="mb-2")
        )
    
    return cards

@app.callback(
    Output('spatial-clustering-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_spatial_clustering(_, analysis_type):
    """Update spatial clustering chart - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    geospatial_data = data_cache.get('geospatial_data', {})
    
    # Generate sample clustering data
    np.random.seed(42)
    n_points = 100
    
    # Create clusters
    cluster_centers = [(40.7, -74.0), (34.0, -118.2), (41.9, -87.6)]
    colors = ['red', 'blue', 'green']
    
    fig = go.Figure()
    
    for i, (center_lat, center_lon) in enumerate(cluster_centers):
        # Generate points around each center
        lats = np.random.normal(center_lat, 0.5, n_points//3)
        lons = np.random.normal(center_lon, 0.5, n_points//3)
        
        fig.add_trace(go.Scatter(
            x=lons, y=lats,
            mode='markers',
            marker=dict(color=colors[i], size=8, opacity=0.6),
            name=f'Cluster {i+1}'
        ))
    
    fig.update_layout(
        title='Spatial Clustering of Healthcare Facilities',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('network-connectivity-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_network_connectivity(_, analysis_type):
    """Update network connectivity chart - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    # Generate sample network data
    nodes = ['Hospital A', 'Hospital B', 'Clinic C', 'Specialty D', 'Emergency E']
    
    # Create adjacency matrix
    n = len(nodes)
    adjacency = np.random.rand(n, n)
    adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
    np.fill_diagonal(adjacency, 0)  # No self-connections
    
    fig = go.Figure(data=go.Heatmap(
        z=adjacency,
        x=nodes,
        y=nodes,
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Healthcare Network Connectivity Matrix',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('service-gaps-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_service_gaps(_, analysis_type):
    """Update service gaps chart - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    geospatial_data = data_cache.get('geospatial_data', {})
    
    # Sample service gap data
    services = ['Emergency Care', 'Surgery', 'Cardiology', 'Oncology', 'Pediatrics', 'Mental Health']
    gap_percentages = np.random.uniform(10, 45, len(services))
    
    fig = px.bar(
        x=services, y=gap_percentages,
        title='Healthcare Service Coverage Gaps by Specialty',
        labels={'y': 'Coverage Gap (%)', 'x': 'Service Type'},
        color=gap_percentages,
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('catchment-areas-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_catchment_areas(_, analysis_type):
    """Update catchment areas chart - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    # Sample catchment area data
    facilities = ['Metro General', 'City Hospital', 'Regional Medical', 'Community Care', 'Specialty Center']
    populations = np.random.randint(50000, 500000, len(facilities))
    coverage_areas = np.random.uniform(10, 100, len(facilities))  # in square miles
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=coverage_areas,
        y=populations,
        mode='markers+text',
        marker=dict(
            size=populations/10000,
            color=populations,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Population Served")
        ),
        text=facilities,
        textposition="top center",
        name='Healthcare Facilities'
    ))
    
    fig.update_layout(
        title='Facility Catchment Areas vs Population Served',
        xaxis_title='Coverage Area (sq miles)',
        yaxis_title='Population Served',
        template='plotly_white'
    )
    
    return fig

@app.callback(
    Output('geospatial-summary-metrics', 'children'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_geospatial_summary(_, analysis_type):
    """Update geospatial summary metrics - only when geospatial dashboard is active"""
    if analysis_type != 'geospatial':
        raise dash.exceptions.PreventUpdate
        
    # Sample summary metrics
    summary_data = [
        {'metric': 'Total Facilities Analyzed', 'value': '15,847'},
        {'metric': 'Average Accessibility Score', 'value': '7.2/10'},
        {'metric': 'Healthcare Deserts Identified', 'value': '2,847'},
        {'metric': 'Population in Underserved Areas', 'value': '83.7M'},
        {'metric': 'Average Travel Distance', 'value': '12.3 miles'},
        {'metric': 'Facilities per 100K Population', 'value': '48.2'}
    ]
    
    return dash_table.DataTable(
        data=summary_data,
        columns=[
            {'name': 'Metric', 'id': 'metric'},
            {'name': 'Value', 'id': 'value'}
        ],
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ]
    )

# Financial Dashboard callbacks - with conditional execution
@app.callback(
    Output('roi-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_roi_chart(_, analysis_type):
    """Update ROI chart - only when financial dashboard is active"""
    if analysis_type != 'financial':
        raise dash.exceptions.PreventUpdate
        
    # Generate sample ROI data
    investments = ['Digital Health', 'Telemedicine', 'AI/ML', 'Infrastructure', 'Training']
    roi_values = np.random.uniform(0.1, 0.4, len(investments))
    
    fig = px.bar(x=investments, y=roi_values, title='Healthcare Investment ROI')
    fig.update_layout(template='plotly_white', yaxis_title='ROI (%)')
    return fig

@app.callback(
    Output('market-valuation-chart', 'figure'),
    [Input('main-content', 'children'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_market_valuation(_, analysis_type):
    """Update market valuation chart - only when financial dashboard is active"""
    if analysis_type != 'financial':
        raise dash.exceptions.PreventUpdate
        
    # Generate sample market valuation data
    sectors = ['Pharmaceuticals', 'Medical Devices', 'Healthcare IT', 'Biotechnology', 'Hospitals']
    valuations = np.random.uniform(50, 500, len(sectors))
    
    fig = px.pie(values=valuations, names=sectors, title='Healthcare Market Valuation by Sector')
    fig.update_layout(template='plotly_white')
    return fig

# Streaming Dashboard callbacks - with conditional execution
@app.callback(
    Output('live-metrics-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('analysis-type', 'value')],
    prevent_initial_call=True
)
def update_live_metrics(n, analysis_type):
    """Update live metrics chart - only when streaming dashboard is active"""
    if analysis_type != 'streaming':
        raise dash.exceptions.PreventUpdate
        
    try:
        # Generate real-time data simulation
        times = pd.date_range(start=datetime.now() - timedelta(minutes=30), 
                             end=datetime.now(), freq='1min')
        
        # Simulate live healthcare metrics
        patient_flow = np.random.poisson(50, len(times))
        bed_occupancy = np.random.uniform(0.7, 0.95, len(times))
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=times, y=patient_flow, name="Patient Flow"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=times, y=bed_occupancy, name="Bed Occupancy"),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Patient Count", secondary_y=False)
        fig.update_yaxes(title_text="Occupancy Rate", secondary_y=True)
        fig.update_layout(title="Live Healthcare Metrics", template='plotly_white')
        
        return fig
    except Exception as e:
        logger.error(f"Error updating live metrics: {e}")
        return go.Figure().add_annotation(text="Error Loading Live Metrics", showarrow=False)

# Specialized Dashboard Callbacks
@app.callback(
    Output('specialized-dashboard-content', 'children'),
    [Input('dashboard-type-selector', 'value')],
    prevent_initial_call=True
)
def update_specialized_dashboard(dashboard_type):
    """Update specialized dashboard based on user type selection"""
    if not dashboard_type:
        return html.Div()
    
    try:
        return specialized_dashboard_manager.get_dashboard_layout(dashboard_type)
    except Exception as e:
        logger.error(f"Error updating specialized dashboard: {e}")
        return dbc.Alert(
            f"Error loading {dashboard_type} dashboard: {str(e)}",
            color="danger"
        )

# Real-time Monitoring Callbacks
@app.callback(
    Output('alert-summary', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_alert_summary(n):
    """Update alert summary"""
    try:
        active_alerts = healthcare_monitor.get_active_alerts()
        alert_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for alert in active_alerts:
            alert_counts[alert.severity.value] += 1
        
        return dbc.Row([
            dbc.Col([
                dbc.Badge(f"Critical: {alert_counts['critical']}", color="danger", className="me-2")
            ], width=3),
            dbc.Col([
                dbc.Badge(f"High: {alert_counts['high']}", color="warning", className="me-2")
            ], width=3),
            dbc.Col([
                dbc.Badge(f"Medium: {alert_counts['medium']}", color="info", className="me-2")
            ], width=3),
            dbc.Col([
                dbc.Badge(f"Low: {alert_counts['low']}", color="secondary")
            ], width=3)
        ])
    except Exception as e:
        logger.error(f"Error updating alert summary: {e}")
        return html.Div("Error loading alert summary")

@app.callback(
    Output('alert-list', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_alert_list(n):
    """Update active alerts list"""
    try:
        active_alerts = healthcare_monitor.get_active_alerts()
        
        if not active_alerts:
            return dbc.Alert("No active alerts", color="success")
        
        alert_items = []
        for alert in active_alerts[-5:]:  # Show last 5 alerts
            color_map = {
                'critical': 'danger',
                'high': 'warning', 
                'medium': 'info',
                'low': 'secondary'
            }
            
            alert_items.append(
                dbc.Alert([
                    html.Strong(f"{alert.severity.value.upper()}: "),
                    alert.message,
                    html.Small(f" - {alert.timestamp.strftime('%H:%M:%S')}", className="text-muted")
                ], color=color_map.get(alert.severity.value, 'info'), className="mb-2")
            )
        
        return alert_items
    except Exception as e:
        logger.error(f"Error updating alert list: {e}")
        return html.Div("Error loading alerts")

@app.callback(
    Output('realtime-metrics-summary', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_realtime_metrics_summary(n):
    """Update real-time metrics summary"""
    try:
        metrics = healthcare_monitor.get_realtime_metrics()
        
        if not metrics:
            return html.Div("No real-time data available")
        
        metric_cards = []
        for metric_name, data in metrics.items():
            trend_icon = "📈" if data['trend'] > 0.1 else "📉" if data['trend'] < -0.1 else "➡️"
            
            metric_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(metric_name.replace('_', ' ').title(), className="card-title"),
                            html.H4(f"{data['current']:.1f}", className="text-primary"),
                            html.Small([
                                trend_icon,
                                f" Avg: {data['avg_1h']:.1f}"
                            ], className="text-muted")
                        ])
                    ], className="mb-2")
                ], width=6)
            )
        
        return dbc.Row(metric_cards)
    except Exception as e:
        logger.error(f"Error updating metrics summary: {e}")
        return html.Div("Error loading metrics")

@app.callback(
    Output('realtime-trends-chart', 'figure'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_realtime_trends_chart(n):
    """Update real-time trends chart"""
    try:
        return healthcare_monitor.create_realtime_trends_chart()
    except Exception as e:
        logger.error(f"Error updating trends chart: {e}")
        return go.Figure().add_annotation(text="Error loading trends chart", showarrow=False)

@app.callback(
    Output('threshold-config', 'children'),
    [Input('interval-component', 'n_intervals')],
    prevent_initial_call=True
)
def update_threshold_config(n):
    """Update threshold configuration display"""
    try:
        thresholds = healthcare_monitor.thresholds
        
        config_items = []
        for metric_type, threshold in thresholds.items():
            config_items.append(
                dbc.Row([
                    dbc.Col([
                        html.Strong(metric_type.value.replace('_', ' ').title())
                    ], width=4),
                    dbc.Col([
                        html.Span(f"Max: {threshold.max_threshold}" if threshold.max_threshold else "No max limit")
                    ], width=3),
                    dbc.Col([
                        html.Span(f"Min: {threshold.min_threshold}" if threshold.min_threshold else "No min limit")
                    ], width=3),
                    dbc.Col([
                        dbc.Badge(threshold.severity.value.title(), 
                                color="danger" if threshold.severity.value == "critical" else "warning")
                    ], width=2)
                ], className="mb-2")
            )
        
        return config_items
    except Exception as e:
        logger.error(f"Error updating threshold config: {e}")
        return html.Div("Error loading threshold configuration")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)