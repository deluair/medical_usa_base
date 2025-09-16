"""
Advanced Machine Learning Models for Healthcare Analytics
"""
import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor
import asyncio

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet

# Suppress warnings
warnings.filterwarnings('ignore')

from config import settings, ML_MODEL_CONFIGS

logger = logging.getLogger(__name__)

class HealthcareCostPredictor:
    """Machine Learning model for predicting healthcare costs"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = None
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for cost prediction"""
        features = df.copy()
        
        # Create age groups
        if 'age' in features.columns:
            features['age_group'] = pd.cut(features['age'], 
                                         bins=[0, 18, 35, 50, 65, 100], 
                                         labels=['0-18', '19-35', '36-50', '51-65', '65+'])
        
        # Encode categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col].astype(str))
            else:
                features[col] = self.label_encoders[col].transform(features[col].astype(str))
        
        # Handle missing values
        features = features.fillna(features.median())
        
        return features
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the cost prediction model"""
        logger.info("Training healthcare cost prediction model...")
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models and select best
        models = {
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_score = float('-inf')
        best_model = None
        
        for name, model in models.items():
            try:
                if name in ['xgboost', 'lightgbm']:
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    predictions = model.predict(X_test_scaled)
                
                score = r2_score(y_test, predictions)
                logger.info(f"{name} R² score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    self.model = model
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        # Calculate metrics
        if self.model:
            if isinstance(self.model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
                y_pred = self.model.predict(X_test)
                self.feature_importance = dict(zip(X_processed.columns, self.model.feature_importances_))
            else:
                y_pred = self.model.predict(X_test_scaled)
                self.feature_importance = dict(zip(X_processed.columns, self.model.feature_importances_))
            
            metrics = {
                'r2_score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            self.is_trained = True
            logger.info(f"Model trained successfully. R² score: {metrics['r2_score']:.4f}")
            return metrics
        
        return {}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make cost predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_features(X)
        
        if isinstance(self.model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
            return self.model.predict(X_processed)
        else:
            X_scaled = self.scaler.transform(X_processed)
            return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_importance or {}
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_importance': self.feature_importance
            }, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_importance = data['feature_importance']
        self.is_trained = True

class HealthcareDemandForecaster:
    """Time series forecasting for healthcare demand"""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        
    def prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare time series data for Prophet"""
        ts_data = df[[date_col, value_col]].copy()
        ts_data.columns = ['ds', 'y']
        ts_data['ds'] = pd.to_datetime(ts_data['ds'])
        ts_data = ts_data.sort_values('ds')
        
        return ts_data
    
    def train(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, float]:
        """Train demand forecasting model"""
        logger.info("Training healthcare demand forecasting model...")
        
        # Prepare data
        ts_data = self.prepare_time_series(df, date_col, value_col)
        
        # Initialize Prophet model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Train model
        self.model.fit(ts_data)
        self.is_trained = True
        
        # Calculate cross-validation metrics
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            df_cv = cross_validation(self.model, initial='730 days', period='180 days', horizon='365 days')
            df_p = performance_metrics(df_cv)
            
            metrics = {
                'mae': df_p['mae'].mean(),
                'mape': df_p['mape'].mean(),
                'rmse': df_p['rmse'].mean()
            }
            
            logger.info(f"Demand forecasting model trained. MAE: {metrics['mae']:.2f}")
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not calculate cross-validation metrics: {e}")
            return {}
    
    def forecast(self, periods: int = 365) -> pd.DataFrame:
        """Generate demand forecast"""
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def get_components(self) -> pd.DataFrame:
        """Get forecast components (trend, seasonality)"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        future = self.model.make_future_dataframe(periods=365)
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'trend', 'yearly', 'weekly', 'monthly']]

class HealthcareRiskAssessment:
    """Risk assessment model for healthcare outcomes"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        
    def prepare_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for risk assessment"""
        features = df.copy()
        
        # Create risk factors
        if 'age' in features.columns:
            features['high_risk_age'] = (features['age'] > 65).astype(int)
        
        if 'chronic_conditions' in features.columns:
            features['multiple_conditions'] = (features['chronic_conditions'] > 2).astype(int)
        
        # Encode categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features[col] = self.label_encoders[col].fit_transform(features[col].astype(str))
            else:
                features[col] = self.label_encoders[col].transform(features[col].astype(str))
        
        return features.fillna(0)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train risk assessment model"""
        logger.info("Training healthcare risk assessment model...")
        
        X_processed = self.prepare_risk_features(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest for risk assessment
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        self.is_trained = True
        logger.info(f"Risk assessment model trained. R² score: {metrics['r2_score']:.4f}")
        
        return metrics
    
    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Predict risk scores"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_risk_features(X)
        X_scaled = self.scaler.transform(X_processed)
        
        return self.model.predict(X_scaled)
    
    def get_risk_factors(self) -> Dict[str, float]:
        """Get most important risk factors"""
        if not self.is_trained:
            return {}
        
        feature_names = self.model.feature_names_in_
        importances = self.model.feature_importances_
        
        return dict(zip(feature_names, importances))

class MonteCarloSimulator:
    """Monte Carlo simulation for healthcare projections"""
    
    def __init__(self):
        self.scenarios = {}
    
    def simulate_cost_projections(self, 
                                base_cost: float, 
                                growth_rate_mean: float, 
                                growth_rate_std: float,
                                years: int = 10,
                                simulations: int = 10000) -> Dict[str, np.ndarray]:
        """Simulate healthcare cost projections using Monte Carlo"""
        
        logger.info(f"Running Monte Carlo simulation with {simulations} scenarios...")
        
        results = {
            'year': np.arange(1, years + 1),
            'scenarios': [],
            'percentiles': {}
        }
        
        for _ in range(simulations):
            scenario_costs = [base_cost]
            
            for year in range(years):
                # Random growth rate from normal distribution
                growth_rate = np.random.normal(growth_rate_mean, growth_rate_std)
                next_cost = scenario_costs[-1] * (1 + growth_rate)
                scenario_costs.append(next_cost)
            
            results['scenarios'].append(scenario_costs[1:])  # Exclude base year
        
        # Calculate percentiles
        scenarios_array = np.array(results['scenarios'])
        results['percentiles'] = {
            'p10': np.percentile(scenarios_array, 10, axis=0),
            'p25': np.percentile(scenarios_array, 25, axis=0),
            'p50': np.percentile(scenarios_array, 50, axis=0),
            'p75': np.percentile(scenarios_array, 75, axis=0),
            'p90': np.percentile(scenarios_array, 90, axis=0),
            'mean': np.mean(scenarios_array, axis=0)
        }
        
        logger.info("Monte Carlo simulation completed")
        return results
    
    def simulate_demand_scenarios(self,
                                base_demand: float,
                                demographic_growth: float,
                                economic_impact: float,
                                years: int = 5,
                                simulations: int = 5000) -> Dict[str, np.ndarray]:
        """Simulate healthcare demand scenarios"""
        
        results = {
            'year': np.arange(1, years + 1),
            'scenarios': [],
            'percentiles': {}
        }
        
        for _ in range(simulations):
            scenario_demand = [base_demand]
            
            for year in range(years):
                # Random factors affecting demand
                demo_factor = np.random.normal(demographic_growth, 0.01)
                econ_factor = np.random.normal(economic_impact, 0.02)
                random_shock = np.random.normal(0, 0.05)  # Random events
                
                total_growth = demo_factor + econ_factor + random_shock
                next_demand = scenario_demand[-1] * (1 + total_growth)
                scenario_demand.append(max(0, next_demand))  # Ensure non-negative
            
            results['scenarios'].append(scenario_demand[1:])
        
        # Calculate percentiles
        scenarios_array = np.array(results['scenarios'])
        results['percentiles'] = {
            'p10': np.percentile(scenarios_array, 10, axis=0),
            'p25': np.percentile(scenarios_array, 25, axis=0),
            'p50': np.percentile(scenarios_array, 50, axis=0),
            'p75': np.percentile(scenarios_array, 75, axis=0),
            'p90': np.percentile(scenarios_array, 90, axis=0),
            'mean': np.mean(scenarios_array, axis=0)
        }
        
        return results

class ModelManager:
    """Manages all ML models and provides unified interface"""
    
    def __init__(self):
        """Initialize model manager with configurations"""
        self.models = {}
        self.model_configs = ML_MODEL_CONFIGS
        self.model_cache_dir = settings.MODEL_CACHE_DIR
        os.makedirs(self.model_cache_dir, exist_ok=True)
        self.cost_predictor = HealthcareCostPredictor()
        self.demand_forecaster = HealthcareDemandForecaster()
        self.risk_assessor = HealthcareRiskAssessment()
        self.monte_carlo = MonteCarloSimulator()
        
    def train_all_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Train all models with provided data"""
        results = {}
        
        # Train cost prediction model
        if 'cost_data' in data:
            cost_data = data['cost_data']
            if 'cost' in cost_data.columns:
                X = cost_data.drop('cost', axis=1)
                y = cost_data['cost']
                results['cost_prediction'] = self.cost_predictor.train(X, y)
        
        # Train demand forecasting model
        if 'demand_data' in data:
            demand_data = data['demand_data']
            if 'date' in demand_data.columns and 'demand' in demand_data.columns:
                results['demand_forecasting'] = self.demand_forecaster.train(
                    demand_data, 'date', 'demand'
                )
        
        # Train risk assessment model
        if 'risk_data' in data:
            risk_data = data['risk_data']
            if 'risk_score' in risk_data.columns:
                X = risk_data.drop('risk_score', axis=1)
                y = risk_data['risk_score']
                results['risk_assessment'] = self.risk_assessor.train(X, y)
        
        return results
    
    def get_all_predictions(self, input_data: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """Get predictions from all trained models"""
        predictions = {}
        
        # Cost predictions
        if 'cost_features' in input_data and self.cost_predictor.is_trained:
            predictions['cost_predictions'] = self.cost_predictor.predict(
                input_data['cost_features']
            )
        
        # Demand forecast
        if self.demand_forecaster.is_trained:
            predictions['demand_forecast'] = self.demand_forecaster.forecast(365)
        
        # Risk scores
        if 'risk_features' in input_data and self.risk_assessor.is_trained:
            predictions['risk_scores'] = self.risk_assessor.predict_risk(
                input_data['risk_features']
            )
        
        return predictions