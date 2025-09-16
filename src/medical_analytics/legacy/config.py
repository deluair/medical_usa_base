"""Configuration settings for the Advanced Medical Analytics Platform"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class Settings:
    """Application settings"""
    # Application
    APP_NAME: str = "Advanced Medical Analytics Platform"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8050
    
    # API Keys (set via environment variables)
    CMS_API_KEY: str = os.getenv("CMS_API_KEY", "")
    BLS_API_KEY: str = os.getenv("BLS_API_KEY", "")
    CENSUS_API_KEY: str = os.getenv("CENSUS_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    
    # API Base URLs - Updated with working real endpoints
    CMS_BASE_URL: str = "https://data.cms.gov/data-api/v1/dataset"
    CMS_COVERAGE_API: str = "https://api.coverage.cms.gov/"
    CMS_MARKETPLACE_API: str = "https://marketplace.api.healthcare.gov/api/v1/"
    BLS_BASE_URL: str = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    CENSUS_BASE_URL: str = "https://api.census.gov/data"
    FRED_BASE_URL: str = "https://api.stlouisfed.org/fred"
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///medical_analytics.db"
    CACHE_TTL: int = 3600  # 1 hour
    CACHE_TIMEOUT: int = 3600  # 1 hour
    LONG_CACHE_TIMEOUT: int = 86400  # 24 hours
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Data Processing
    MAX_WORKERS: int = 4
    BATCH_SIZE: int = 1000
    CHUNK_SIZE: int = 10000
    
    # Machine Learning
    MODEL_CACHE_DIR: str = "./models"
    RANDOM_SEED: int = 42
    TRAIN_TEST_SPLIT: float = 0.8
    
    # Geospatial Settings
    DEFAULT_CRS: str = "EPSG:4326"
    BUFFER_DISTANCE: float = 50000  # 50km in meters
    
    # API Rate Limits (requests per minute)
    DEFAULT_RATE_LIMIT: int = 60
    CMS_RATE_LIMIT: int = 1000
    BLS_RATE_LIMIT: int = 500

# Global settings instance
settings = Settings()

# Data source configurations - Updated with real working APIs
DATA_SOURCES = {
    "cms_coverage": {
        "url": f"{settings.CMS_COVERAGE_API}",
        "description": "CMS Medicare Coverage Database - Real API (no key required)",
        "update_frequency": "daily",
        "rate_limit": 10000  # requests per second
    },
    "cms_marketplace": {
        "url": f"{settings.CMS_MARKETPLACE_API}",
        "description": "CMS Marketplace API - Health Insurance Plans",
        "update_frequency": "daily",
        "endpoints": {
            "plans_search": "plans/search",
            "counties": "counties/by/zip/",
            "drugs": "drugs/autocomplete"
        }
    },
    "cms_open_payments": {
        "url": f"{settings.CMS_BASE_URL}/029c119f-f79c-49be-9100-344d31d10344/data",
        "description": "CMS Open Payments Data - Real API endpoint",
        "update_frequency": "monthly"
    },
    "bls_employment": {
        "url": f"{settings.BLS_BASE_URL}",
        "series_ids": {
            "healthcare_employment": "CEU6562000001",  # Healthcare and Social Assistance Employment
            "hospitals_employment": "CEU6562200001",   # Hospitals Employment
            "nursing_employment": "CEU6562300001",     # Nursing and Residential Care Employment
            "ambulatory_employment": "CEU6562100001"   # Ambulatory Health Care Services Employment
        },
        "description": "BLS Healthcare Employment Data - Real API",
        "update_frequency": "monthly",
        "requires_registration": True
    },
    "census_population": {
        "url": f"{settings.CENSUS_BASE_URL}/2021/acs/acs5",
        "description": "Census Population Data",
        "update_frequency": "yearly"
    }
}

# API rate limits
API_RATE_LIMITS = {
    "cms": settings.CMS_RATE_LIMIT,
    "bls": settings.BLS_RATE_LIMIT,
    "census": settings.DEFAULT_RATE_LIMIT,
    "fred": settings.DEFAULT_RATE_LIMIT
}

# Machine Learning model configurations
ML_MODEL_CONFIGS = {
    "cost_predictor": {
        "model_type": "xgboost",
        "features": ["population", "gdp_per_capita", "age_median", "hospital_beds"],
        "target": "healthcare_spending_per_capita",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        }
    },
    "demand_forecaster": {
        "model_type": "prophet",
        "seasonality_mode": "multiplicative",
        "yearly_seasonality": True,
        "weekly_seasonality": False,
        "daily_seasonality": False
    },
    "risk_assessor": {
        "model_type": "lightgbm",
        "features": ["financial_metrics", "operational_metrics", "market_metrics"],
        "target": "risk_score",
        "hyperparameters": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9
        }
    }
}