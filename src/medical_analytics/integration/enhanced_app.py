"""
Enhanced Medical Analytics Application
Integrates all advanced features: Monte Carlo simulations, nuanced analytics, 
3D visualizations, AI insights, and real-time simulations
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Any, Optional
import asyncio
import threading

# Import our advanced modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from simulations.advanced_monte_carlo import AdvancedMonteCarloEngine, ScenarioType, UncertaintyLevel
from analytics.nuanced_analytics import NuancedHealthcareAnalytics, EquityDimension, AnalysisComplexity
from visualizations.advanced_3d_viz import Advanced3DHealthcareVisualizer, VisualizationType, ColorScheme, VisualizationConfig
from ai.insights_engine import AIHealthcareInsightsEngine, InsightType, ConfidenceLevel
from simulations.realtime_engine import RealTimeSimulationEngine, UpdateFrequency, HealthcareScenarios

# Import existing modules
from data.data_fetcher import MedicalDataFetcher
from data.advanced_data_fetcher import AdvancedMedicalDataFetcher
from data.data_pipeline import ETLPipeline

logger = logging.getLogger(__name__)

class EnhancedMedicalAnalyticsApp:
    """Enhanced Medical Analytics Application with all advanced features"""
    
    def __init__(self):
        """Initialize the enhanced application"""
        
        # Initialize core components
        self.data_fetcher = MedicalDataFetcher()
        self.advanced_data_fetcher = AdvancedMedicalDataFetcher()
        self.etl_pipeline = ETLPipeline()
        
        # Initialize advanced engines
        self.monte_carlo_engine = AdvancedMonteCarloEngine()
        self.nuanced_analytics = NuancedHealthcareAnalytics()
        self.viz_engine = Advanced3DHealthcareVisualizer()
        self.insights_engine = AIHealthcareInsightsEngine()
        self.realtime_engine = RealTimeSimulationEngine()
        
        # Application state
        self.current_data = {}
        self.analysis_results = {}
        self.simulation_results = {}
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                dbc.themes.BOOTSTRAP,
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"
            ],
            suppress_callback_exceptions=True
        )
        
        # Setup layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Load initial data
        self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial data for the application"""
        
        try:
            # Load cached data from ETL pipeline
            self.current_data = self.etl_pipeline.load_all_data()
            
            if self.current_data:
                logger.info("Initial data loaded successfully")
                
                # Generate initial insights
                self._generate_initial_insights()
                
            else:
                logger.warning("No cached data available")
                
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
    
    def _generate_initial_insights(self):
        """Generate initial AI insights from loaded data"""
        
        try:
            if 'cms_data' in self.current_data and not self.current_data['cms_data'].empty:
                
                # Generate insights
                insights = self.insights_engine.generate_comprehensive_insights(
                    self.current_data['cms_data'],
                    analysis_depth=AnalysisDepth.COMPREHENSIVE
                )
                
                self.analysis_results['initial_insights'] = insights
                logger.info("Initial insights generated successfully")
                
        except Exception as e:
            logger.error(f"Error generating initial insights: {e}")
    
    def _setup_layout(self):
        """Setup the enhanced application layout"""
        
        self.app.layout = dbc.Container([
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.I(className="fas fa-heartbeat me-3"),
                            "Enhanced Medical Analytics Platform"
                        ], className="text-center text-primary mb-2"),
                        html.P(
                            "Advanced Healthcare Analytics with AI Insights, Real-time Simulations, and 3D Visualizations",
                            className="text-center text-muted lead"
                        )
                    ])
                ])
            ], className="mb-4"),
            
            # Navigation Tabs
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab(label="📊 Dashboard Overview", tab_id="dashboard"),
                        dbc.Tab(label="🎲 Monte Carlo Simulations", tab_id="monte_carlo"),
                        dbc.Tab(label="🔍 Nuanced Analytics", tab_id="nuanced_analytics"),
                        dbc.Tab(label="🌐 3D Visualizations", tab_id="3d_viz"),
                        dbc.Tab(label="🤖 AI Insights", tab_id="ai_insights"),
                        dbc.Tab(label="⚡ Real-time Simulations", tab_id="realtime")
                    ], id="main-tabs", active_tab="dashboard")
                ])
            ], className="mb-4"),
            
            # Tab Content
            html.Div(id="tab-content"),
            
            # Footer
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P([
                        "Enhanced Medical Analytics Platform © 2024 | ",
                        html.A("Documentation", href="#", className="text-decoration-none"),
                        " | ",
                        html.A("API Reference", href="#", className="text-decoration-none")
                    ], className="text-center text-muted small")
                ])
            ])
            
        ], fluid=True, className="px-4")
    
    def _setup_callbacks(self):
        """Setup all application callbacks"""
        
        @self.app.callback(
            Output("tab-content", "children"),
            [Input("main-tabs", "active_tab")]
        )
        def render_tab_content(active_tab):
            """Render content based on active tab"""
            
            if active_tab == "dashboard":
                return self._render_dashboard_tab()
            elif active_tab == "monte_carlo":
                return self._render_monte_carlo_tab()
            elif active_tab == "nuanced_analytics":
                return self._render_nuanced_analytics_tab()
            elif active_tab == "3d_viz":
                return self._render_3d_viz_tab()
            elif active_tab == "ai_insights":
                return self._render_ai_insights_tab()
            elif active_tab == "realtime":
                return self._render_realtime_tab()
            else:
                return html.Div("Tab not found")
        
        # Monte Carlo simulation callbacks
        @self.app.callback(
            [Output("monte-carlo-results", "children"),
             Output("monte-carlo-chart", "figure")],
            [Input("run-monte-carlo", "n_clicks")],
            [State("scenario-type", "value"),
             State("uncertainty-level", "value"),
             State("num-simulations", "value")]
        )
        def run_monte_carlo_simulation(n_clicks, scenario_type, uncertainty_level, num_simulations):
            """Run Monte Carlo simulation"""
            
            if not n_clicks:
                return html.Div("Click 'Run Simulation' to start"), {}
            
            try:
                # Convert string values to enums
                scenario = ScenarioType(scenario_type)
                uncertainty = UncertaintyLevel(uncertainty_level)
                
                # Run simulation
                results = self.monte_carlo_engine.run_scenario_simulation(
                    scenario=scenario,
                    uncertainty_level=uncertainty,
                    num_simulations=num_simulations or 1000
                )
                
                # Create results summary
                summary = dbc.Card([
                    dbc.CardHeader("Monte Carlo Simulation Results"),
                    dbc.CardBody([
                        html.H5(f"Scenario: {scenario.value.title()}"),
                        html.P(f"Uncertainty Level: {uncertainty.value.title()}"),
                        html.P(f"Simulations: {num_simulations or 1000:,}"),
                        html.Hr(),
                        html.H6("Key Metrics:"),
                        html.Ul([
                            html.Li(f"Mean Cost Impact: ${results['cost_impact']['mean']:,.2f}"),
                            html.Li(f"95% Confidence Interval: ${results['cost_impact']['ci_lower']:,.2f} - ${results['cost_impact']['ci_upper']:,.2f}"),
                            html.Li(f"Risk of Exceeding Budget: {results['risk_metrics']['budget_risk']:.1%}")
                        ])
                    ])
                ])
                
                # Create visualization
                fig = go.Figure()
                
                # Add histogram of cost impacts
                fig.add_trace(go.Histogram(
                    x=results['raw_data']['cost_impacts'],
                    nbinsx=50,
                    name="Cost Impact Distribution",
                    opacity=0.7
                ))
                
                # Add confidence interval lines
                fig.add_vline(
                    x=results['cost_impact']['ci_lower'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="95% CI Lower"
                )
                
                fig.add_vline(
                    x=results['cost_impact']['ci_upper'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="95% CI Upper"
                )
                
                fig.update_layout(
                    title="Monte Carlo Simulation Results",
                    xaxis_title="Cost Impact ($)",
                    yaxis_title="Frequency",
                    height=500
                )
                
                return summary, fig
                
            except Exception as e:
                error_msg = dbc.Alert(f"Error running simulation: {str(e)}", color="danger")
                return error_msg, {}
        
        # Nuanced analytics callbacks
        @self.app.callback(
            [Output("nuanced-results", "children"),
             Output("equity-chart", "figure")],
            [Input("run-nuanced-analysis", "n_clicks")],
            [State("equity-dimension", "value"),
             State("analysis-complexity", "value")]
        )
        def run_nuanced_analysis(n_clicks, equity_dimension, analysis_complexity):
            """Run nuanced analytics"""
            
            if not n_clicks or not self.current_data.get('cms_data'):
                return html.Div("Click 'Run Analysis' to start"), {}
            
            try:
                # Convert string values to enums
                dimension = EquityDimension(equity_dimension)
                complexity = AnalysisComplexity(analysis_complexity)
                
                # Run equity analysis
                results = self.nuanced_analytics.analyze_health_equity(
                    self.current_data['cms_data'],
                    dimension=dimension,
                    complexity=complexity
                )
                
                # Create results summary
                summary = dbc.Card([
                    dbc.CardHeader("Nuanced Analytics Results"),
                    dbc.CardBody([
                        html.H5(f"Equity Dimension: {dimension.value.title()}"),
                        html.P(f"Analysis Complexity: {complexity.value.title()}"),
                        html.Hr(),
                        html.H6("Key Findings:"),
                        html.Ul([
                            html.Li(f"Equity Score: {results.equity_score:.3f}"),
                            html.Li(f"Disparity Index: {results.disparity_index:.3f}"),
                            html.Li(f"Segments Analyzed: {len(results.segment_analysis)}")
                        ]),
                        html.H6("Recommendations:"),
                        html.Ul([html.Li(rec) for rec in results.recommendations[:3]])
                    ])
                ])
                
                # Create equity visualization
                fig = px.bar(
                    x=list(results.segment_analysis.keys()),
                    y=list(results.segment_analysis.values()),
                    title="Health Equity Analysis by Segment",
                    labels={'x': 'Segments', 'y': 'Equity Score'}
                )
                
                fig.update_layout(height=400)
                
                return summary, fig
                
            except Exception as e:
                error_msg = dbc.Alert(f"Error running analysis: {str(e)}", color="danger")
                return error_msg, {}
        
        # 3D visualization callbacks
        @self.app.callback(
            Output("3d-visualization", "figure"),
            [Input("viz-type", "value"),
             Input("color-scheme", "value")]
        )
        def update_3d_visualization(viz_type, color_scheme):
            """Update 3D visualization"""
            
            if not self.current_data.get('cms_data'):
                return {}
            
            try:
                # Convert string values to enums
                viz_type_enum = VisualizationType(viz_type)
                color_scheme_enum = ColorScheme(color_scheme)
                
                # Generate 3D visualization
                config = VisualizationConfig(
                    title="3D Healthcare Landscape",
                    color_scheme=color_scheme_enum,
                    three_dimensional=True
                )
                
                # Generate sample data for visualization
                sample_data = pd.DataFrame({
                    'age_numeric': np.random.uniform(20, 80, 100),
                    'income_numeric': np.random.uniform(20000, 120000, 100),
                    'health_score': np.random.uniform(40, 100, 100)
                })
                
                if viz_type_enum == VisualizationType.HEALTH_LANDSCAPE:
                    fig = self.viz_engine.create_3d_health_landscape(
                        sample_data, 'age_numeric', 'income_numeric', 'health_score', config
                    )
                elif viz_type_enum == VisualizationType.NETWORK_3D:
                    # Create sample network data
                    nodes_data = pd.DataFrame({
                        'id': range(10),
                        'label': [f'Node_{i}' for i in range(10)],
                        'x': np.random.uniform(-1, 1, 10),
                        'y': np.random.uniform(-1, 1, 10),
                        'z': np.random.uniform(-1, 1, 10),
                        'size': np.random.uniform(5, 20, 10)
                    })
                    edges_data = pd.DataFrame({
                        'source': np.random.choice(10, 15),
                        'target': np.random.choice(10, 15),
                        'weight': np.random.uniform(0.1, 1.0, 15)
                    })
                    fig = self.viz_engine.create_network_3d_visualization(
                        nodes_data, edges_data, config
                    )
                else:
                    # Default to gradient heatmap
                    fig = self.viz_engine.create_gradient_heatmap(
                        sample_data, 'age_numeric', 'income_numeric', 'health_score', config
                    )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating 3D visualization: {e}")
                return {}
        
        # AI insights callbacks
        @self.app.callback(
            Output("ai-insights-content", "children"),
            [Input("generate-insights", "n_clicks")],
            [State("insight-type", "value"),
             State("analysis-depth", "value")]
        )
        def generate_ai_insights(n_clicks, insight_type, analysis_depth):
            """Generate AI insights"""
            
            if not n_clicks or not self.current_data.get('cms_data'):
                return html.Div("Click 'Generate Insights' to start")
            
            try:
                # Convert string values to enums
                insight_type_enum = InsightType(insight_type)
                depth_enum = AnalysisDepth(analysis_depth)
                
                # Generate insights
                insights = self.insights_engine.generate_insights(
                    self.current_data['cms_data'],
                    insight_type=insight_type_enum,
                    analysis_depth=depth_enum
                )
                
                # Create insights display
                insights_cards = []
                
                for insight in insights[:5]:  # Show top 5 insights
                    
                    # Determine card color based on confidence
                    if insight.confidence > 0.8:
                        color = "success"
                    elif insight.confidence > 0.6:
                        color = "warning"
                    else:
                        color = "secondary"
                    
                    card = dbc.Card([
                        dbc.CardHeader([
                            html.H6(insight.title, className="mb-0"),
                            dbc.Badge(f"{insight.confidence:.1%} confidence", color=color, className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.P(insight.description),
                            html.Small(f"Impact: {insight.impact}", className="text-muted")
                        ])
                    ], className="mb-3")
                    
                    insights_cards.append(card)
                
                return insights_cards
                
            except Exception as e:
                return dbc.Alert(f"Error generating insights: {str(e)}", color="danger")
    
    def _render_dashboard_tab(self):
        """Render the main dashboard tab"""
        
        return dbc.Row([
            
            # Key Metrics Cards
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("🏥", className="text-primary"),
                                html.H5("Healthcare Spending"),
                                html.H3("$4.1T", className="text-success"),
                                html.Small("Annual US Healthcare Expenditure")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("👥", className="text-info"),
                                html.H5("Healthcare Workers"),
                                html.H3("22.2M", className="text-info"),
                                html.Small("Total Healthcare Employment")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("📈", className="text-warning"),
                                html.H5("Growth Rate"),
                                html.H3("5.8%", className="text-warning"),
                                html.Small("Annual Healthcare Growth")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("🎯", className="text-danger"),
                                html.H5("Efficiency Score"),
                                html.H3("78%", className="text-danger"),
                                html.Small("System Efficiency Rating")
                            ])
                        ])
                    ], width=3)
                ], className="mb-4"),
                
                # Feature Overview
                dbc.Card([
                    dbc.CardHeader("🚀 Enhanced Platform Features"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("🎲 Monte Carlo Simulations"),
                                html.P("Advanced scenario modeling with uncertainty analysis", className="small text-muted")
                            ], width=4),
                            dbc.Col([
                                html.H6("🔍 Nuanced Analytics"),
                                html.P("Demographic segmentation and health equity analysis", className="small text-muted")
                            ], width=4),
                            dbc.Col([
                                html.H6("🌐 3D Visualizations"),
                                html.P("Interactive 3D health landscapes and network maps", className="small text-muted")
                            ], width=4)
                        ]),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                html.H6("🤖 AI Insights"),
                                html.P("Machine learning-powered trend detection and forecasting", className="small text-muted")
                            ], width=6),
                            dbc.Col([
                                html.H6("⚡ Real-time Simulations"),
                                html.P("Live parameter adjustment with instant feedback", className="small text-muted")
                            ], width=6)
                        ])
                    ])
                ])
                
            ], width=12)
        ])
    
    def _render_monte_carlo_tab(self):
        """Render Monte Carlo simulations tab"""
        
        return dbc.Row([
            
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎲 Monte Carlo Simulation Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Scenario Type"),
                                dcc.Dropdown(
                                    id="scenario-type",
                                    options=[
                                        {"label": "Pandemic Response", "value": "pandemic"},
                                        {"label": "Policy Change", "value": "policy_change"},
                                        {"label": "Market Disruption", "value": "market_disruption"},
                                        {"label": "Technology Adoption", "value": "technology_adoption"}
                                    ],
                                    value="pandemic"
                                )
                            ], width=6),
                            
                            dbc.Col([
                                dbc.Label("Uncertainty Level"),
                                dcc.Dropdown(
                                    id="uncertainty-level",
                                    options=[
                                        {"label": "Low", "value": "low"},
                                        {"label": "Medium", "value": "medium"},
                                        {"label": "High", "value": "high"},
                                        {"label": "Extreme", "value": "extreme"}
                                    ],
                                    value="medium"
                                )
                            ], width=6)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Number of Simulations"),
                                dbc.Input(
                                    id="num-simulations",
                                    type="number",
                                    value=1000,
                                    min=100,
                                    max=10000,
                                    step=100
                                )
                            ], width=6),
                            
                            dbc.Col([
                                html.Br(),
                                dbc.Button(
                                    "Run Simulation",
                                    id="run-monte-carlo",
                                    color="primary",
                                    size="lg",
                                    className="w-100"
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=4),
            
            # Results
            dbc.Col([
                html.Div(id="monte-carlo-results", className="mb-4"),
                dcc.Graph(id="monte-carlo-chart")
            ], width=8)
        ])
    
    def _render_nuanced_analytics_tab(self):
        """Render nuanced analytics tab"""
        
        return dbc.Row([
            
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔍 Nuanced Analytics Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Equity Dimension"),
                                dcc.Dropdown(
                                    id="equity-dimension",
                                    options=[
                                        {"label": "Racial", "value": "racial"},
                                        {"label": "Economic", "value": "economic"},
                                        {"label": "Geographic", "value": "geographic"},
                                        {"label": "Age", "value": "age"},
                                        {"label": "Gender", "value": "gender"}
                                    ],
                                    value="racial"
                                )
                            ], width=12)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Analysis Complexity"),
                                dcc.Dropdown(
                                    id="analysis-complexity",
                                    options=[
                                        {"label": "Basic", "value": "basic"},
                                        {"label": "Intermediate", "value": "intermediate"},
                                        {"label": "Advanced", "value": "advanced"},
                                        {"label": "Expert", "value": "expert"}
                                    ],
                                    value="intermediate"
                                )
                            ], width=12)
                        ], className="mb-3"),
                        
                        dbc.Button(
                            "Run Analysis",
                            id="run-nuanced-analysis",
                            color="info",
                            size="lg",
                            className="w-100"
                        )
                    ])
                ])
            ], width=4),
            
            # Results
            dbc.Col([
                html.Div(id="nuanced-results", className="mb-4"),
                dcc.Graph(id="equity-chart")
            ], width=8)
        ])
    
    def _render_3d_viz_tab(self):
        """Render 3D visualizations tab"""
        
        return dbc.Row([
            
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🌐 3D Visualization Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Visualization Type"),
                                dcc.Dropdown(
                                    id="viz-type",
                                    options=[
                                        {"label": "Health Landscape", "value": "health_landscape"},
                                        {"label": "Gradient Heatmap", "value": "gradient_heatmap"},
                                        {"label": "3D Network Map", "value": "network_3d"},
                                        {"label": "Multi-dimensional", "value": "multi_dimensional"}
                                    ],
                                    value="health_landscape"
                                )
                            ], width=12)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Color Scheme"),
                                dcc.Dropdown(
                                    id="color-scheme",
                                    options=[
                                        {"label": "Viridis", "value": "viridis"},
                                        {"label": "Plasma", "value": "plasma"},
                                        {"label": "Inferno", "value": "inferno"},
                                        {"label": "Turbo", "value": "turbo"}
                                    ],
                                    value="viridis"
                                )
                            ], width=12)
                        ])
                    ])
                ])
            ], width=3),
            
            # 3D Visualization
            dbc.Col([
                dcc.Graph(
                    id="3d-visualization",
                    style={"height": "600px"}
                )
            ], width=9)
        ])
    
    def _render_ai_insights_tab(self):
        """Render AI insights tab"""
        
        return dbc.Row([
            
            # Controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🤖 AI Insights Controls"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Insight Type"),
                                dcc.Dropdown(
                                    id="insight-type",
                                    options=[
                                        {"label": "Trend Analysis", "value": "trend"},
                                        {"label": "Anomaly Detection", "value": "anomaly"},
                                        {"label": "Pattern Recognition", "value": "pattern"},
                                        {"label": "Predictive Forecast", "value": "predictive"}
                                    ],
                                    value="trend"
                                )
                            ], width=12)
                        ], className="mb-3"),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Analysis Depth"),
                                dcc.Dropdown(
                                    id="analysis-depth",
                                    options=[
                                        {"label": "Quick", "value": "quick"},
                                        {"label": "Standard", "value": "standard"},
                                        {"label": "Deep", "value": "deep"},
                                        {"label": "Comprehensive", "value": "comprehensive"}
                                    ],
                                    value="standard"
                                )
                            ], width=12)
                        ], className="mb-3"),
                        
                        dbc.Button(
                            "Generate Insights",
                            id="generate-insights",
                            color="success",
                            size="lg",
                            className="w-100"
                        )
                    ])
                ])
            ], width=4),
            
            # Insights Display
            dbc.Col([
                html.Div(id="ai-insights-content")
            ], width=8)
        ])
    
    def _render_realtime_tab(self):
        """Render real-time simulations tab"""
        
        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("⚡ Real-time Simulation Engine"),
                    dbc.CardBody([
                        html.P("The real-time simulation engine runs on a separate dashboard for optimal performance."),
                        html.P("Features include:"),
                        html.Ul([
                            html.Li("Live parameter adjustment"),
                            html.Li("Instant feedback and updates"),
                            html.Li("WebSocket-based real-time communication"),
                            html.Li("Interactive scenario modeling"),
                            html.Li("Event logging and monitoring")
                        ]),
                        html.Hr(),
                        dbc.Button(
                            "Launch Real-time Dashboard",
                            href="http://127.0.0.1:8051",
                            target="_blank",
                            color="warning",
                            size="lg",
                            className="w-100"
                        ),
                        html.Small("Opens in a new tab on port 8051", className="text-muted d-block mt-2")
                    ])
                ])
            ], width=12)
        ])
    
    def run(self, host="127.0.0.1", port=8050, debug=False):
        """Run the enhanced application"""
        
        print("🚀 Enhanced Medical Analytics Platform")
        print("=" * 50)
        print(f"🌐 Main Dashboard: http://{host}:{port}")
        print(f"⚡ Real-time Engine: http://{host}:8051")
        print("\n📋 Available Features:")
        print("  🎲 Monte Carlo Simulations")
        print("  🔍 Nuanced Analytics")
        print("  🌐 3D Visualizations")
        print("  🤖 AI Insights")
        print("  ⚡ Real-time Simulations")
        print("\n🎯 Enhanced with shades and nuances for comprehensive healthcare analytics")
        
        # Start real-time engine in background
        realtime_thread = threading.Thread(
            target=lambda: self.realtime_engine.run_dashboard(host="127.0.0.1", port=8051, debug=False),
            daemon=True
        )
        realtime_thread.start()
        
        # Run main application
        self.app.run_server(host=host, port=port, debug=debug)

# Create and run the enhanced application
if __name__ == "__main__":
    app = EnhancedMedicalAnalyticsApp()
    app.run(debug=False)