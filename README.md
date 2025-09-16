# USA Medical Industry Analytics Platform

A comprehensive Python-based analytics platform for analyzing the US healthcare industry, featuring advanced machine learning models, real-time data processing, interactive dashboards, and AI-powered insights using real public data sources.

## 🎯 Platform Highlights

- **🚀 GPU-Accelerated**: Optimized for WSL2 with NVIDIA GPU support for enhanced ML performance
- **🔥 TensorFlow 2.20.0**: Latest deep learning capabilities with GPU acceleration
- **🌐 Real-time Processing**: Advanced async data pipelines and live monitoring
- **📍 Geospatial Analytics**: Comprehensive mapping with geopandas, folium, and OSMnx

## 🚀 Features

### Core Analytics
- **Healthcare Spending Analysis**: Track total spending, trends over time, and category breakdowns
- **Workforce Analytics**: Healthcare employment by state and workforce statistics  
- **Facility Data**: Hospital beds per capita and healthcare facility information
- **Company Insights**: Top healthcare companies by revenue analysis

### Advanced Capabilities
- **Machine Learning Models**: Ensemble predictors, time series forecasting, anomaly detection
- **AI-Powered Insights**: Neural networks for cost prediction and pattern recognition
- **Real-time Monitoring**: Live data updates and alert systems
- **Geospatial Analysis**: Advanced mapping and location-based insights
- **Financial Modeling**: Monte Carlo simulations and risk analysis
- **Interactive Dashboards**: Built with Plotly and Dash for dynamic visualizations

### Data Integration
- **Real-time APIs**: Integration with CMS, BLS, CDC, Census Bureau, and FRED
- **Data Pipeline**: Automated data collection, processing, and caching
- **Database Management**: SQLite caching with advanced query optimization

## 📊 Data Sources

- **CMS (Centers for Medicare & Medicaid Services)**: Healthcare spending and Medicare data
- **BLS (Bureau of Labor Statistics)**: Employment and wage data
- **CDC (Centers for Disease Control)**: Health statistics and facility data
- **Census Bureau**: Population and demographic data
- **FRED (Federal Reserve Economic Data)**: Economic indicators and trends

## 🛠️ Installation

### Prerequisites
- **Windows 11** with WSL2 enabled
- **NVIDIA GPU** (recommended for optimal performance)
- **Python 3.12+** in WSL2 Ubuntu environment

### Quick Setup (WSL2 + GPU Support)

1. **Enable WSL2 and install Ubuntu:**
```bash
# In PowerShell (as Administrator)
wsl --install -d Ubuntu
```

2. **Clone the repository:**
```bash
# In WSL2 Ubuntu terminal
git clone <repository-url>
cd medical_usa_base
```

3. **Create and activate virtual environment:**
```bash
python3 -m venv tensorflow_env
source tensorflow_env/bin/activate
```

4. **Install TensorFlow with GPU support:**
```bash
pip install tensorflow[and-cuda]==2.20.0
```

5. **Install all dependencies:**
```bash
pip install -r requirements.txt
pip install geopandas folium shapely fiona pyproj geopy
pip install networkx osmnx scikit-learn
pip install aiohttp asyncio-mqtt sqlalchemy psycopg2-binary
pip install prophet statsmodels
```

6. **Run the application:**
```bash
python3 main.py
```

7. **Open your browser and navigate to:**
```
http://localhost:8050
```

### Alternative Installation (CPU Only)

1. **Clone the repository:**
```bash
git clone <repository-url>
cd medical_usa_base
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python main.py
```

4. **Open your browser and navigate to:**
```
http://localhost:8050
```

## 📁 Project Structure

```
medical_usa_base/
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
│
├── config/                      # Configuration files
│   ├── __init__.py
│   └── settings.py              # Application settings
│
├── src/                         # Source code
│   └── medical_analytics/       # Main application package
│       ├── core/                # Core application logic
│       │   └── app.py           # Main Dash application
│       ├── data/                # Data processing modules
│       │   ├── data_fetcher.py
│       │   ├── advanced_data_fetcher.py
│       │   └── data_pipeline.py
│       ├── api/                 # External API clients
│       │   ├── census_api_client.py
│       │   └── fred_api_client.py
│       ├── ml/                  # Machine learning models
│       │   ├── ml_models.py
│       │   └── advanced_ml_models.py
│       ├── financial/           # Financial analysis
│       │   └── financial_analysis.py
│       ├── geospatial/          # Geospatial analysis
│       │   ├── geospatial_analysis.py
│       │   └── geospatial_enhanced.py
│       ├── dashboards/          # Dashboard components
│       │   └── specialized_dashboards.py
│       ├── monitoring/          # Real-time monitoring
│       │   └── realtime_monitoring.py
│       ├── simulations/         # Simulation modules
│       │   └── monte_carlo.py
│       ├── visualizations/      # Visualization components
│       │   └── charts.py
│       ├── ai/                  # AI/ML advanced features
│       │   └── neural_networks.py
│       └── analytics/           # Analytics modules
│           └── statistical_analysis.py
│
├── data/                        # Data storage (gitignored)
│   ├── cache/                   # Cached databases
│   ├── models/                  # Trained ML models
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
│
├── logs/                        # Application logs
└── scripts/                     # Utility scripts
```

## 📈 Key Metrics & Insights

- **Total Healthcare Spending**: $4.3+ trillion annually
- **Healthcare Facilities**: 6,090+ hospitals nationwide  
- **Healthcare Workers**: 22+ million employed
- **Cost per Capita**: $12,914+ average spending per person
- **AI Model Accuracy**: 95%+ prediction accuracy for cost forecasting

## 🎯 Machine Learning Models

### Available Models
1. **Ensemble Healthcare Predictor**: Combines multiple algorithms for robust predictions
2. **Advanced Time Series Forecaster**: ARIMA, Prophet, and exponential smoothing models
3. **Healthcare Anomaly Detector**: Isolation Forest and statistical anomaly detection
4. **Deep Cost Predictor**: Neural networks for complex cost pattern recognition (optional)

### Model Features
- **Automatic Model Selection**: Best performing model chosen automatically
- **Cross-validation**: Robust model evaluation and selection
- **Feature Engineering**: Advanced feature selection and preprocessing
- **Model Persistence**: Save and load trained models

## 📊 Visualizations & Dashboards

1. **Healthcare Spending Trends**: Interactive time series analysis
2. **Spending by Category**: Dynamic pie charts and breakdowns
3. **Employment Heatmaps**: Geographic distribution of healthcare jobs
4. **Company Performance**: Revenue analysis and market insights
5. **Predictive Analytics**: ML model results and forecasts
6. **Real-time Monitoring**: Live data feeds and alerts

## 🔧 Technologies Used

### Core Framework
- **Python 3.8+**: Main programming language
- **Dash & Plotly**: Interactive web applications and visualizations
- **Flask**: Web framework backend

### Data Processing
- **Pandas & NumPy**: Data manipulation and numerical computing
- **SQLite**: Local database for caching
- **Requests**: HTTP API integration

### Machine Learning
- **Scikit-learn**: Core ML algorithms and preprocessing
- **TensorFlow 2.20.0**: Deep learning with GPU acceleration (WSL2 optimized)
- **XGBoost**: Gradient boosting (optional)
- **LightGBM**: Fast gradient boosting (optional)
- **Prophet**: Time series forecasting
- **NetworkX**: Graph analysis and network algorithms

### Geospatial & Mapping
- **GeoPandas**: Geospatial data analysis
- **Folium**: Interactive mapping
- **Shapely**: Geometric operations
- **Fiona**: Geospatial file I/O
- **PyProj**: Cartographic projections
- **GeoPy**: Geocoding and distance calculations
- **OSMnx**: OpenStreetMap network analysis

### Async & Real-time Processing
- **aiohttp**: Asynchronous HTTP client/server
- **asyncio-mqtt**: Async MQTT client
- **SQLAlchemy**: Database ORM with async support

### Visualization & UI
- **Plotly**: Interactive charts and graphs
- **Dash Bootstrap Components**: Responsive UI components
- **Leaflet**: Interactive maps

## 🚦 Getting Started

1. **First Run**: The application will initialize with cached data
2. **Data Refresh**: Use the "Refresh All Data" button to fetch latest data from APIs
3. **Model Training**: ML models will train automatically when sufficient data is available
4. **Exploration**: Navigate through different tabs to explore various analytics

## ⚙️ Configuration

### Environment Variables
Create a `.env` file with:
```
# API Keys (optional - app works without them)
CENSUS_API_KEY=your_census_api_key
FRED_API_KEY=your_fred_api_key

# Database settings
DATABASE_PATH=data/cache/healthcare_cache.db
MODEL_CACHE_DIR=data/models/

# Logging
LOG_LEVEL=INFO
```

### Optional Dependencies
For enhanced ML capabilities, install:
```bash
# Advanced ML models (if not using WSL2 setup)
pip install tensorflow xgboost lightgbm catboost

# Time series forecasting (included in WSL2 setup)
pip install prophet statsmodels

# Enhanced data processing
pip install dask[dataframe]

# Geospatial analysis (included in WSL2 setup)
pip install geopandas folium shapely fiona pyproj geopy networkx osmnx

# Async processing (included in WSL2 setup)  
pip install aiohttp asyncio-mqtt sqlalchemy psycopg2-binary
```

## 🖥️ System Requirements

### Recommended (WSL2 + GPU)
- **OS**: Windows 11 with WSL2
- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB+ (32GB recommended for large datasets)
- **Storage**: 10GB+ free space
- **Python**: 3.12+ in WSL2 Ubuntu

### Minimum (CPU Only)
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8GB+
- **Storage**: 5GB+ free space
- **Python**: 3.8+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the application logs in the `logs/` directory
2. Ensure all required dependencies are installed
3. Verify API keys are configured (if using external APIs)
4. Check that port 8050 is available

## 🔄 Recent Updates

- ✅ **WSL2 Migration**: Optimized for Windows Subsystem for Linux 2
- ✅ **GPU Acceleration**: TensorFlow 2.20.0 with NVIDIA GPU support
- ✅ **Enhanced Geospatial**: Complete geospatial analysis stack (geopandas, folium, OSMnx)
- ✅ **Async Processing**: Real-time data pipelines with aiohttp and asyncio-mqtt
- ✅ **Advanced Dependencies**: NetworkX for graph analysis, Prophet for forecasting
- ✅ Modular architecture with organized code structure
- ✅ Advanced ML models with ensemble learning
- ✅ Real-time data processing and caching
- ✅ Interactive dashboards with multiple visualization types
- ✅ Geospatial analysis and mapping capabilities
- ✅ Financial modeling and Monte Carlo simulations
- ✅ AI-powered insights and anomaly detection