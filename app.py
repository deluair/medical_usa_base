import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import requests
from datetime import datetime
import dash_bootstrap_components as dbc
from data_fetcher import MedicalDataFetcher

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "USA Medical Industry Analytics"

# Initialize data fetcher
data_fetcher = MedicalDataFetcher()

# Define the layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("USA Medical Industry Analytics Dashboard", 
                   className="text-center mb-4 text-primary"),
            html.P("Comprehensive analysis of healthcare spending, facilities, workforce, and industry trends",
                   className="text-center text-muted mb-5")
        ])
    ]),
    
    # Key Metrics Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Total Healthcare Spending", className="card-title"),
                    html.H2(id="total-spending", className="text-success"),
                    html.P("Annual healthcare expenditure", className="card-text")
                ])
            ], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Healthcare Facilities", className="card-title"),
                    html.H2(id="total-facilities", className="text-info"),
                    html.P("Hospitals and clinics nationwide", className="card-text")
                ])
            ], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Healthcare Workers", className="card-title"),
                    html.H2(id="total-workers", className="text-warning"),
                    html.P("Total healthcare employment", className="card-text")
                ])
            ], className="mb-4")
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Average Cost per Capita", className="card-title"),
                    html.H2(id="cost-per-capita", className="text-danger"),
                    html.P("Healthcare spending per person", className="card-text")
                ])
            ], className="mb-4")
        ], width=3)
    ]),
    
    # Charts Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Healthcare Spending Trends")),
                dbc.CardBody([
                    dcc.Graph(id="spending-trends-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Healthcare Spending by Category")),
                dbc.CardBody([
                    dcc.Graph(id="spending-category-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Healthcare Employment by State")),
                dbc.CardBody([
                    dcc.Graph(id="employment-map")
                ])
            ])
        ], width=12)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Top Healthcare Companies by Revenue")),
                dbc.CardBody([
                    dcc.Graph(id="companies-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Hospital Beds per 1000 Population")),
                dbc.CardBody([
                    dcc.Graph(id="hospital-beds-chart")
                ])
            ])
        ], width=6)
    ]),
    
    # Footer
    html.Hr(),
    html.P(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
           className="text-center text-muted mt-4")
], fluid=True)

# Callbacks for updating charts and metrics
@app.callback(
    [Output('total-spending', 'children'),
     Output('total-facilities', 'children'),
     Output('total-workers', 'children'),
     Output('cost-per-capita', 'children')],
    [Input('spending-trends-chart', 'id')]  # Dummy input to trigger on load
)
def update_metrics(_):
    metrics = data_fetcher.get_key_metrics()
    return (
        f"${metrics['total_spending']:,.0f}B",
        f"{metrics['total_facilities']:,}",
        f"{metrics['total_workers']:,.0f}M",
        f"${metrics['cost_per_capita']:,.0f}"
    )

@app.callback(
    Output('spending-trends-chart', 'figure'),
    [Input('spending-trends-chart', 'id')]
)
def update_spending_trends(_):
    df = data_fetcher.get_spending_trends()
    fig = px.line(df, x='year', y='spending', 
                  title='Healthcare Spending Over Time',
                  labels={'spending': 'Spending (Billions USD)', 'year': 'Year'})
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('spending-category-chart', 'figure'),
    [Input('spending-category-chart', 'id')]
)
def update_spending_categories(_):
    df = data_fetcher.get_spending_by_category()
    fig = px.pie(df, values='amount', names='category',
                 title='Healthcare Spending by Category')
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('employment-map', 'figure'),
    [Input('employment-map', 'id')]
)
def update_employment_map(_):
    df = data_fetcher.get_employment_by_state()
    fig = px.choropleth(df, locations='state_code', values='employment',
                        locationmode='USA-states',
                        title='Healthcare Employment by State',
                        color_continuous_scale='Blues')
    fig.update_layout(geo_scope='usa', template='plotly_white')
    return fig

@app.callback(
    Output('companies-chart', 'figure'),
    [Input('companies-chart', 'id')]
)
def update_companies_chart(_):
    df = data_fetcher.get_top_companies()
    fig = px.bar(df, x='revenue', y='company', orientation='h',
                 title='Top Healthcare Companies by Revenue',
                 labels={'revenue': 'Revenue (Billions USD)', 'company': 'Company'})
    fig.update_layout(template='plotly_white')
    return fig

@app.callback(
    Output('hospital-beds-chart', 'figure'),
    [Input('hospital-beds-chart', 'id')]
)
def update_hospital_beds_chart(_):
    df = data_fetcher.get_hospital_beds_data()
    fig = px.bar(df, x='state', y='beds_per_1000',
                 title='Hospital Beds per 1000 Population by State',
                 labels={'beds_per_1000': 'Beds per 1000 Population', 'state': 'State'})
    fig.update_layout(template='plotly_white', xaxis_tickangle=-45)
    return fig

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)