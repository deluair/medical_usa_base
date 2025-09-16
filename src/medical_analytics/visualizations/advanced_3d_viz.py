"""
Advanced 3D Healthcare Visualization Engine
Sophisticated visualizations with 3D plots, gradient heatmaps, interactive scenario modeling,
and immersive data exploration with nuanced visual storytelling
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.colors import n_colors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class VisualizationType(Enum):
    """Types of advanced visualizations"""
    SURFACE_3D = "surface_3d"
    VOLUME_3D = "volume_3d"
    NETWORK_3D = "network_3d"
    HEATMAP_GRADIENT = "heatmap_gradient"
    INTERACTIVE_SCENARIO = "interactive_scenario"
    ANIMATED_TIMELINE = "animated_timeline"
    MULTI_DIMENSIONAL = "multi_dimensional"
    IMMERSIVE_DASHBOARD = "immersive_dashboard"

class ColorScheme(Enum):
    """Advanced color schemes for healthcare data"""
    HEALTH_GRADIENT = "health_gradient"
    RISK_SPECTRUM = "risk_spectrum"
    EQUITY_DIVERGING = "equity_diverging"
    CLINICAL_COOL = "clinical_cool"
    WARM_CARE = "warm_care"
    SEVERITY_SCALE = "severity_scale"
    DEMOGRAPHIC_RAINBOW = "demographic_rainbow"

@dataclass
class VisualizationConfig:
    """Configuration for advanced visualizations"""
    title: str
    color_scheme: ColorScheme
    interactive: bool = True
    animation: bool = False
    three_dimensional: bool = False
    gradient_intensity: float = 1.0
    opacity_levels: List[float] = None
    custom_annotations: List[str] = None

class Advanced3DHealthcareVisualizer:
    """Advanced 3D visualization engine for healthcare analytics"""
    
    def __init__(self):
        self.color_schemes = self._initialize_color_schemes()
        self.figure_cache = {}
        
    def _initialize_color_schemes(self) -> Dict[ColorScheme, Dict]:
        """Initialize sophisticated color schemes for healthcare visualizations"""
        return {
            ColorScheme.HEALTH_GRADIENT: {
                'colors': ['#FF6B6B', '#FFE66D', '#4ECDC4', '#45B7D1', '#96CEB4'],
                'description': 'Red (poor) to Green (excellent) health gradient'
            },
            ColorScheme.RISK_SPECTRUM: {
                'colors': ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#8B0000'],
                'description': 'Low risk (green) to high risk (dark red)'
            },
            ColorScheme.EQUITY_DIVERGING: {
                'colors': ['#8E44AD', '#3498DB', '#F8F9FA', '#E74C3C', '#C0392B'],
                'description': 'Diverging scale for equity analysis'
            },
            ColorScheme.CLINICAL_COOL: {
                'colors': ['#E8F4FD', '#B3D9F2', '#7FB8D3', '#4C96B4', '#1F7A8C'],
                'description': 'Cool clinical blues and teals'
            },
            ColorScheme.WARM_CARE: {
                'colors': ['#FFF5E6', '#FFE0B3', '#FFCC80', '#FFB74D', '#FF9800'],
                'description': 'Warm caring oranges and yellows'
            },
            ColorScheme.SEVERITY_SCALE: {
                'colors': ['#E8F5E8', '#C8E6C9', '#A5D6A7', '#81C784', '#66BB6A', 
                          '#4CAF50', '#43A047', '#388E3C', '#2E7D32', '#1B5E20'],
                'description': 'Fine-grained severity scale'
            },
            ColorScheme.DEMOGRAPHIC_RAINBOW: {
                'colors': ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', 
                          '#99CCFF', '#FFB366', '#B3B3FF'],
                'description': 'Distinct colors for demographic groups'
            }
        }
    
    def create_3d_health_landscape(self, data: pd.DataFrame, 
                                 x_col: str, y_col: str, z_col: str,
                                 config: VisualizationConfig) -> go.Figure:
        """Create a 3D landscape visualization of health metrics"""
        
        # Prepare data for 3D surface
        x_unique = sorted(data[x_col].unique())
        y_unique = sorted(data[y_col].unique())
        
        # Create meshgrid
        X, Y = np.meshgrid(x_unique, y_unique)
        Z = np.zeros_like(X, dtype=float)
        
        # Fill Z values
        for i, y_val in enumerate(y_unique):
            for j, x_val in enumerate(x_unique):
                mask = (data[x_col] == x_val) & (data[y_col] == y_val)
                if mask.any():
                    Z[i, j] = data.loc[mask, z_col].mean()
                else:
                    # Interpolate missing values
                    Z[i, j] = np.nan
        
        # Handle NaN values with interpolation
        from scipy.interpolate import griddata
        
        # Get valid points
        valid_mask = ~np.isnan(Z)
        if valid_mask.sum() > 3:  # Need at least 3 points for interpolation
            points = np.column_stack((X[valid_mask], Y[valid_mask]))
            values = Z[valid_mask]
            
            # Interpolate missing values
            nan_mask = np.isnan(Z)
            if nan_mask.any():
                nan_points = np.column_stack((X[nan_mask], Y[nan_mask]))
                Z[nan_mask] = griddata(points, values, nan_points, method='linear', fill_value=0)
        
        # Create 3D surface plot
        colors = self.color_schemes[config.color_scheme]['colors']
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=colors,
            opacity=0.8,
            contours={
                "x": {"show": True, "start": X.min(), "end": X.max(), "size": (X.max()-X.min())/10},
                "y": {"show": True, "start": Y.min(), "end": Y.max(), "size": (Y.max()-Y.min())/10},
                "z": {"show": True, "start": Z.min(), "end": Z.max(), "size": (Z.max()-Z.min())/10}
            },
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                         f'<b>{y_col}</b>: %{{y}}<br>' +
                         f'<b>{z_col}</b>: %{{z:.2f}}<extra></extra>'
        )])
        
        # Add scatter points for actual data
        fig.add_trace(go.Scatter3d(
            x=data[x_col],
            y=data[y_col],
            z=data[z_col],
            mode='markers',
            marker=dict(
                size=3,
                color=data[z_col],
                colorscale=colors,
                opacity=0.6
            ),
            name='Data Points',
            hovertemplate=f'<b>{x_col}</b>: %{{x}}<br>' +
                         f'<b>{y_col}</b>: %{{y}}<br>' +
                         f'<b>{z_col}</b>: %{{z:.2f}}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title(),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                bgcolor='rgba(240,240,240,0.1)'
            ),
            width=900,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig
    
    def create_gradient_heatmap(self, data: pd.DataFrame, 
                              x_col: str, y_col: str, value_col: str,
                              config: VisualizationConfig) -> go.Figure:
        """Create sophisticated gradient heatmap with smooth transitions"""
        
        # Create pivot table
        heatmap_data = data.pivot_table(
            values=value_col, 
            index=y_col, 
            columns=x_col, 
            aggfunc='mean'
        )
        
        # Apply Gaussian smoothing for gradient effect
        from scipy.ndimage import gaussian_filter
        smoothed_data = gaussian_filter(heatmap_data.fillna(0), sigma=config.gradient_intensity)
        
        # Create custom colorscale
        colors = self.color_schemes[config.color_scheme]['colors']
        n_colors_needed = len(colors)
        colorscale = []
        
        for i, color in enumerate(colors):
            colorscale.append([i / (n_colors_needed - 1), color])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=smoothed_data,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='<b>%{y}</b><br>' +
                         '<b>%{x}</b><br>' +
                         f'<b>{value_col}</b>: %{{z:.2f}}<extra></extra>',
            colorbar=dict(
                title=value_col.replace('_', ' ').title(),
                titleside='right',
                thickness=15,
                len=0.7
            )
        ))
        
        # Add contour lines for depth
        fig.add_trace(go.Contour(
            z=smoothed_data,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            showscale=False,
            contours=dict(
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=10, color='white')
            ),
            line=dict(width=1, color='rgba(255,255,255,0.3)'),
            hoverinfo='skip'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            width=800,
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_interactive_scenario_dashboard(self, scenarios: Dict[str, pd.DataFrame],
                                           config: VisualizationConfig) -> go.Figure:
        """Create interactive dashboard for scenario comparison"""
        
        # Create subplots
        n_scenarios = len(scenarios)
        cols = min(3, n_scenarios)
        rows = (n_scenarios + cols - 1) // cols
        
        subplot_titles = list(scenarios.keys())
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{'type': 'surface'} for _ in range(cols)] for _ in range(rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = self.color_schemes[config.color_scheme]['colors']
        
        for idx, (scenario_name, scenario_data) in enumerate(scenarios.items()):
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Create surface for each scenario
            if len(scenario_data) > 0:
                # Assume scenario_data has x, y, z columns
                x_col = scenario_data.columns[0]
                y_col = scenario_data.columns[1]
                z_col = scenario_data.columns[2] if len(scenario_data.columns) > 2 else scenario_data.columns[1]
                
                # Create meshgrid for surface
                x_unique = sorted(scenario_data[x_col].unique())
                y_unique = sorted(scenario_data[y_col].unique())
                
                if len(x_unique) > 1 and len(y_unique) > 1:
                    X, Y = np.meshgrid(x_unique, y_unique)
                    Z = np.zeros_like(X, dtype=float)
                    
                    for i, y_val in enumerate(y_unique):
                        for j, x_val in enumerate(x_unique):
                            mask = (scenario_data[x_col] == x_val) & (scenario_data[y_col] == y_val)
                            if mask.any():
                                Z[i, j] = scenario_data.loc[mask, z_col].mean()
                    
                    fig.add_trace(
                        go.Surface(
                            x=X, y=Y, z=Z,
                            colorscale=colors,
                            opacity=0.7,
                            showscale=False,
                            name=scenario_name
                        ),
                        row=row, col=col
                    )
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 20, 'color': '#2C3E50'}
            },
            height=300 * rows + 100,
            width=1200,
            showlegend=False
        )
        
        return fig
    
    def create_multi_dimensional_visualization(self, data: pd.DataFrame,
                                             dimensions: List[str],
                                             config: VisualizationConfig) -> go.Figure:
        """Create multi-dimensional visualization with parallel coordinates and 3D scatter"""
        
        # Create subplot with parallel coordinates and 3D scatter
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{'type': 'parcoords'}],
                   [{'type': 'scatter3d'}]],
            subplot_titles=['Parallel Coordinates View', '3D Scatter View'],
            vertical_spacing=0.15,
            row_heights=[0.4, 0.6]
        )
        
        # Parallel coordinates plot
        colors = self.color_schemes[config.color_scheme]['colors']
        
        # Prepare dimensions for parallel coordinates
        parcoords_dimensions = []
        for dim in dimensions[:6]:  # Limit to 6 dimensions for readability
            if dim in data.columns:
                parcoords_dimensions.append(dict(
                    range=[data[dim].min(), data[dim].max()],
                    label=dim.replace('_', ' ').title(),
                    values=data[dim]
                ))
        
        if len(parcoords_dimensions) > 0:
            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color=data[dimensions[0]] if dimensions[0] in data.columns else range(len(data)),
                        colorscale=colors,
                        showscale=True,
                        colorbar=dict(
                            title=dimensions[0].replace('_', ' ').title(),
                            y=0.8,
                            len=0.3
                        )
                    ),
                    dimensions=parcoords_dimensions
                ),
                row=1, col=1
            )
        
        # 3D scatter plot
        if len(dimensions) >= 3:
            x_dim, y_dim, z_dim = dimensions[:3]
            
            if all(dim in data.columns for dim in [x_dim, y_dim, z_dim]):
                # Color by fourth dimension if available
                color_dim = dimensions[3] if len(dimensions) > 3 and dimensions[3] in data.columns else z_dim
                
                fig.add_trace(
                    go.Scatter3d(
                        x=data[x_dim],
                        y=data[y_dim],
                        z=data[z_dim],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=data[color_dim],
                            colorscale=colors,
                            opacity=0.7,
                            colorbar=dict(
                                title=color_dim.replace('_', ' ').title(),
                                y=0.3,
                                len=0.3
                            )
                        ),
                        text=[f'{x_dim}: {x}<br>{y_dim}: {y}<br>{z_dim}: {z}<br>{color_dim}: {c}'
                              for x, y, z, c in zip(data[x_dim], data[y_dim], data[z_dim], data[color_dim])],
                        hovertemplate='%{text}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            height=800,
            width=1000,
            showlegend=False
        )
        
        return fig
    
    def create_animated_timeline_visualization(self, data: pd.DataFrame,
                                            time_col: str, value_cols: List[str],
                                            config: VisualizationConfig) -> go.Figure:
        """Create animated timeline visualization showing evolution over time"""
        
        # Prepare data for animation
        time_points = sorted(data[time_col].unique())
        
        # Create frames for animation
        frames = []
        
        for time_point in time_points:
            frame_data = data[data[time_col] == time_point]
            
            frame_traces = []
            colors = self.color_schemes[config.color_scheme]['colors']
            
            for i, col in enumerate(value_cols):
                if col in frame_data.columns:
                    color = colors[i % len(colors)]
                    
                    # Create different visualizations based on data
                    if len(frame_data) > 1:
                        # Line plot for multiple points
                        frame_traces.append(
                            go.Scatter(
                                x=frame_data.index,
                                y=frame_data[col],
                                mode='lines+markers',
                                name=col.replace('_', ' ').title(),
                                line=dict(color=color, width=3),
                                marker=dict(size=8, color=color)
                            )
                        )
                    else:
                        # Bar plot for single values
                        frame_traces.append(
                            go.Bar(
                                x=[col.replace('_', ' ').title()],
                                y=[frame_data[col].iloc[0]],
                                name=col.replace('_', ' ').title(),
                                marker_color=color
                            )
                        )
            
            frames.append(go.Frame(
                data=frame_traces,
                name=str(time_point)
            ))
        
        # Create initial figure
        initial_data = data[data[time_col] == time_points[0]]
        initial_traces = []
        
        colors = self.color_schemes[config.color_scheme]['colors']
        
        for i, col in enumerate(value_cols):
            if col in initial_data.columns:
                color = colors[i % len(colors)]
                
                if len(initial_data) > 1:
                    initial_traces.append(
                        go.Scatter(
                            x=initial_data.index,
                            y=initial_data[col],
                            mode='lines+markers',
                            name=col.replace('_', ' ').title(),
                            line=dict(color=color, width=3),
                            marker=dict(size=8, color=color)
                        )
                    )
                else:
                    initial_traces.append(
                        go.Bar(
                            x=[col.replace('_', ' ').title()],
                            y=[initial_data[col].iloc[0]],
                            name=col.replace('_', ' ').title(),
                            marker_color=color
                        )
                    )
        
        fig = go.Figure(
            data=initial_traces,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(tp)], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': str(tp),
                        'method': 'animate'
                    }
                    for tp in time_points
                ],
                'active': 0,
                'currentvalue': {'prefix': f'{time_col}: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }],
            width=900,
            height=600
        )
        
        return fig
    
    def create_network_3d_visualization(self, nodes: pd.DataFrame, edges: pd.DataFrame,
                                      config: VisualizationConfig) -> go.Figure:
        """Create 3D network visualization for healthcare relationships"""
        
        # Create 3D layout for nodes
        n_nodes = len(nodes)
        
        # Use spring layout in 3D
        np.random.seed(42)
        
        # Simple 3D positioning (could be enhanced with proper 3D layout algorithms)
        theta = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
        phi = np.random.uniform(0, np.pi, n_nodes)
        r = np.random.uniform(0.5, 2.0, n_nodes)
        
        x_pos = r * np.sin(phi) * np.cos(theta)
        y_pos = r * np.sin(phi) * np.sin(theta)
        z_pos = r * np.cos(phi)
        
        # Create edge traces
        edge_traces = []
        colors = self.color_schemes[config.color_scheme]['colors']
        
        for _, edge in edges.iterrows():
            if 'source' in edge and 'target' in edge:
                source_idx = edge['source']
                target_idx = edge['target']
                
                if source_idx < len(x_pos) and target_idx < len(x_pos):
                    edge_traces.extend([
                        go.Scatter3d(
                            x=[x_pos[source_idx], x_pos[target_idx], None],
                            y=[y_pos[source_idx], y_pos[target_idx], None],
                            z=[z_pos[source_idx], z_pos[target_idx], None],
                            mode='lines',
                            line=dict(color=colors[0], width=2),
                            hoverinfo='none',
                            showlegend=False
                        )
                    ])
        
        # Create node trace
        node_colors = nodes.get('value', range(len(nodes)))
        
        node_trace = go.Scatter3d(
            x=x_pos,
            y=y_pos,
            z=z_pos,
            mode='markers+text',
            marker=dict(
                size=nodes.get('size', [10] * len(nodes)),
                color=node_colors,
                colorscale=colors,
                opacity=0.8,
                colorbar=dict(
                    title='Node Value',
                    thickness=15,
                    len=0.7
                )
            ),
            text=nodes.get('label', [f'Node {i}' for i in range(len(nodes))]),
            textposition='middle center',
            hovertemplate='<b>%{text}</b><br>Value: %{marker.color}<extra></extra>',
            name='Nodes'
        )
        
        # Combine all traces
        all_traces = edge_traces + [node_trace]
        
        fig = go.Figure(data=all_traces)
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.title,
                'x': 0.5,
                'font': {'size': 18, 'color': '#2C3E50'}
            },
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='rgba(240,240,240,0.1)'
            ),
            showlegend=False,
            width=800,
            height=700,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        
        return fig

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample healthcare data
    np.random.seed(42)
    n_samples = 1000
    
    # Sample data for 3D landscape
    sample_data = pd.DataFrame({
        'age_group': np.random.choice(['18-30', '31-50', '51-70', '70+'], n_samples),
        'income_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'health_score': np.random.normal(75, 15, n_samples),
        'access_score': np.random.normal(70, 20, n_samples),
        'satisfaction': np.random.normal(7.5, 1.5, n_samples),
        'year': np.random.choice([2020, 2021, 2022, 2023], n_samples)
    })
    
    # Initialize visualizer
    visualizer = Advanced3DHealthcareVisualizer()
    
    # Create 3D health landscape
    config_3d = VisualizationConfig(
        title="3D Healthcare Landscape: Age vs Income vs Health Score",
        color_scheme=ColorScheme.HEALTH_GRADIENT,
        three_dimensional=True
    )
    
    # Convert categorical to numerical for 3D plot
    sample_data['age_numeric'] = sample_data['age_group'].map({
        '18-30': 25, '31-50': 40, '51-70': 60, '70+': 80
    })
    sample_data['income_numeric'] = sample_data['income_level'].map({
        'Low': 30000, 'Medium': 60000, 'High': 100000
    })
    
    fig_3d = visualizer.create_3d_health_landscape(
        sample_data, 'age_numeric', 'income_numeric', 'health_score', config_3d
    )
    
    print("Advanced 3D Healthcare Visualizations Created Successfully!")
    print("Features include:")
    print("- 3D surface landscapes with health metrics")
    print("- Gradient heatmaps with smooth transitions")
    print("- Interactive scenario comparison dashboards")
    print("- Multi-dimensional parallel coordinates")
    print("- Animated timeline visualizations")
    print("- 3D network relationship mapping")