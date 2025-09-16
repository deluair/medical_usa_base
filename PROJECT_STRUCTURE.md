# Project Structure - Medical Analytics Platform

## Organized Directory Layout

```
medical_usa_base/
├── README.md                    # Project documentation
├── main.py                      # Application entry point
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
├── .gitignore                   # Git ignore rules
│
├── config/                      # Configuration files
│   ├── __init__.py
│   └── settings.py              # Application settings
│
├── src/                         # Source code
│   └── medical_analytics/       # Main application package
│       ├── __init__.py
│       ├── core/                # Core application logic
│       │   ├── __init__.py
│       │   └── app.py           # Main Dash application
│       ├── data/                # Data processing modules
│       │   ├── __init__.py
│       │   ├── data_fetcher.py
│       │   ├── advanced_data_fetcher.py
│       │   └── data_pipeline.py
│       ├── api/                 # External API clients
│       │   ├── __init__.py
│       │   ├── census_api_client.py
│       │   └── fred_api_client.py
│       ├── ml/                  # Machine learning models
│       │   ├── __init__.py
│       │   ├── ml_models.py
│       │   └── advanced_ml_models.py
│       ├── financial/           # Financial analysis
│       │   ├── __init__.py
│       │   └── financial_analysis.py
│       ├── geospatial/          # Geospatial analysis
│       │   ├── __init__.py
│       │   ├── geospatial_analysis.py
│       │   └── geospatial_enhanced.py
│       ├── dashboards/          # Dashboard components
│       │   ├── __init__.py
│       │   └── specialized_dashboards.py
│       ├── monitoring/          # Real-time monitoring
│       │   ├── __init__.py
│       │   └── realtime_monitoring.py
│       ├── integration/         # Integration modules
│       │   ├── __init__.py
│       │   └── enhanced_app.py
│       ├── simulations/         # Simulation modules
│       │   ├── __init__.py
│       │   └── monte_carlo.py
│       ├── visualizations/      # Visualization components
│       │   ├── __init__.py
│       │   └── charts.py
│       ├── ai/                  # AI/ML advanced features
│       │   ├── __init__.py
│       │   └── neural_networks.py
│       ├── analytics/           # Analytics modules
│       │   ├── __init__.py
│       │   └── statistical_analysis.py
│       └── legacy/              # Legacy/deprecated files
│           ├── __init__.py
│           └── config.py        # Old config file
│
├── data/                        # Data storage (gitignored)
│   ├── cache/                   # Cached data and databases
│   │   ├── healthcare_cache.db
│   │   └── medical_data_cache.db
│   ├── models/                  # Trained ML models
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
│
├── logs/                        # Application logs
│   ├── __init__.py
│   └── data_download.log
│
├── scripts/                     # Utility scripts
│   └── download_data.py
│
├── tests/                       # Test files
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
│
└── docs/                        # Documentation
```

## Key Improvements Made

### 1. **Clean Root Directory**
- Moved all Python modules to appropriate subdirectories
- Only essential files remain in root (main.py, README.md, requirements.txt)

### 2. **Organized Data Storage**
- Created `data/` directory with subdirectories:
  - `cache/` - Database files and cached data
  - `models/` - Trained ML models
  - `raw/` - Raw data files
  - `processed/` - Processed data files

### 3. **Structured Source Code**
- All modules properly organized under `src/medical_analytics/`
- Logical grouping by functionality:
  - `data/` - Data processing and fetching
  - `api/` - External API clients
  - `ml/` - Machine learning components
  - `financial/` - Financial analysis
  - `geospatial/` - Geospatial analysis
  - `core/` - Main application logic

### 4. **Proper Package Structure**
- Added `__init__.py` files to all directories
- Proper import paths maintained
- Legacy files moved to `legacy/` directory

### 5. **Logging and Configuration**
- Dedicated `logs/` directory for application logs
- Centralized configuration in `config/` directory

## Benefits

- **Maintainability**: Clear separation of concerns
- **Scalability**: Easy to add new modules
- **Collaboration**: Standard Python project structure
- **Deployment**: Clean structure for containerization
- **Testing**: Organized test structure
- **Documentation**: Dedicated docs directory

## Usage

The application entry point remains the same:
```bash
python main.py
```

All import paths have been maintained, so existing functionality continues to work seamlessly.
