"""
Advanced Machine Learning Models for Healthcare Analytics
Enhanced with deep learning, ensemble methods, and specialized healthcare models
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
import joblib
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Enhanced ML Libraries
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor, IsolationForest
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    TimeSeriesSplit, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    LabelEncoder, OneHotEncoder, PolynomialFeatures
)
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, RFE

# Advanced ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from prophet import Prophet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from config.settings import settings, ML_MODEL_CONFIGS

logger = logging.getLogger(__name__)

class DeepHealthcareCostPredictor:
    """Advanced deep learning model for healthcare cost prediction"""
    
    def __init__(self, input_dim: int = 20, hidden_dims: List[int] = [128, 64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=input_dim)
        self.is_trained = False
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build deep neural network for cost prediction"""
        model = keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(self.hidden_dims[0], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(self.hidden_dims[1], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(self.hidden_dims[2], activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Advanced feature engineering for healthcare cost prediction"""
        features = df.copy()
        
        # Create interaction features
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            for i, col1 in enumerate(numeric_cols[:5]):  # Limit to avoid explosion
                for col2 in numeric_cols[i+1:6]:
                    features[f'{col1}_{col2}_interaction'] = features[col1] * features[col2]
        
        # Polynomial features for key variables
        if 'age' in features.columns:
            features['age_squared'] = features['age'] ** 2
            features['age_cubed'] = features['age'] ** 3
        
        # Time-based features
        if 'date' in features.columns:
            features['date'] = pd.to_datetime(features['date'])
            features['year'] = features['date'].dt.year
            features['month'] = features['date'].dt.month
            features['quarter'] = features['date'].dt.quarter
            features['day_of_year'] = features['date'].dt.dayofyear
        
        # Handle categorical variables with target encoding
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'date':
                # Simple label encoding for now
                le = LabelEncoder()
                features[col] = le.fit_transform(features[col].astype(str))
        
        # Fill missing values
        features = features.fillna(features.median())
        
        # Select only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        return numeric_features.values
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the deep learning model"""
        logger.info("Training deep healthcare cost prediction model...")
        
        # Prepare features
        X_processed = self.prepare_features(X)
        
        # Feature selection
        X_selected = self.feature_selector.fit_transform(X_processed, y)
        self.input_dim = X_selected.shape[1]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42
        )
        
        # Build and train model
        self.model = self.build_model()
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train, verbose=0)
        test_pred = self.model.predict(X_test, verbose=0)
        
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred)
        }
        
        self.is_trained = True
        logger.info(f"Model trained successfully. Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_processed = self.prepare_features(X)
        X_selected = self.feature_selector.transform(X_processed)
        X_scaled = self.scaler.transform(X_selected)
        
        predictions = self.model.predict(X_scaled, verbose=0)
        return predictions.flatten()

class EnsembleHealthcarePredictor:
    """Ensemble model combining multiple algorithms for robust predictions"""
    
    def __init__(self):
        self.models = {}
        self.meta_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize base models for ensemble"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=200, depth=6, learning_rate=0.1,
                random_seed=42, verbose=False
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
        }
        
        # Meta-learner for stacking
        self.meta_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
        )
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train ensemble model with cross-validation"""
        logger.info("Training ensemble healthcare prediction model...")
        
        self.initialize_models()
        
        # Prepare features
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y.values, test_size=0.2, random_state=42
        )
        
        # Train base models and collect predictions for meta-learning
        meta_features_train = np.zeros((X_train.shape[0], len(self.models)))
        meta_features_test = np.zeros((X_test.shape[0], len(self.models)))
        
        model_scores = {}
        
        for i, (name, model) in enumerate(self.models.items()):
            logger.info(f"Training {name}...")
            
            # Cross-validation for meta-features
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            model_scores[name] = cv_scores.mean()
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Generate meta-features
            meta_features_train[:, i] = model.predict(X_train)
            meta_features_test[:, i] = model.predict(X_test)
        
        # Train meta-model
        self.meta_model.fit(meta_features_train, y_train)
        
        # Final predictions
        final_pred_train = self.meta_model.predict(meta_features_train)
        final_pred_test = self.meta_model.predict(meta_features_test)
        
        # Evaluate ensemble
        metrics = {
            'train_r2': r2_score(y_train, final_pred_train),
            'test_r2': r2_score(y_test, final_pred_test),
            'train_mse': mean_squared_error(y_train, final_pred_train),
            'test_mse': mean_squared_error(y_test, final_pred_test),
            'model_scores': model_scores
        }
        
        self.is_trained = True
        logger.info(f"Ensemble trained successfully. Test R²: {metrics['test_r2']:.4f}")
        
        return metrics
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for ensemble model"""
        features = df.copy()
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
        
        # Fill missing values
        features = features.fillna(features.median())
        
        return features.select_dtypes(include=[np.number]).values
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        X_processed = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_processed)
        
        # Get predictions from all base models
        meta_features = np.zeros((X_scaled.shape[0], len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            meta_features[:, i] = model.predict(X_scaled)
        
        # Final prediction from meta-model
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions

class AdvancedTimeSeriesForecaster:
    """Advanced time series forecasting with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.is_trained = False
        self.data = None
        
    def train(self, df: pd.DataFrame, date_col: str, value_col: str) -> Dict[str, Any]:
        """Train multiple time series models and select the best"""
        logger.info("Training advanced time series forecasting models...")
        
        # Prepare data
        self.data = df[[date_col, value_col]].copy()
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data = self.data.sort_values(date_col).reset_index(drop=True)
        
        # Split data for validation
        split_idx = int(len(self.data) * 0.8)
        train_data = self.data[:split_idx]
        test_data = self.data[split_idx:]
        
        model_scores = {}
        
        # Prophet model
        try:
            prophet_data = train_data.rename(columns={date_col: 'ds', value_col: 'y'})
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            prophet_model.fit(prophet_data)
            
            # Forecast on test period
            future = prophet_model.make_future_dataframe(periods=len(test_data))
            forecast = prophet_model.predict(future)
            prophet_pred = forecast['yhat'].iloc[-len(test_data):].values
            
            prophet_score = r2_score(test_data[value_col], prophet_pred)
            model_scores['prophet'] = prophet_score
            self.models['prophet'] = prophet_model
            
        except Exception as e:
            logger.warning(f"Prophet model failed: {e}")
        
        # ARIMA model
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(train_data[value_col], order=(2, 1, 2))
            arima_fitted = arima_model.fit()
            
            arima_pred = arima_fitted.forecast(steps=len(test_data))
            arima_score = r2_score(test_data[value_col], arima_pred)
            model_scores['arima'] = arima_score
            self.models['arima'] = arima_fitted
            
        except Exception as e:
            logger.warning(f"ARIMA model failed: {e}")
        
        # Exponential Smoothing
        try:
            exp_smooth_model = ExponentialSmoothing(
                train_data[value_col],
                trend='add',
                seasonal='add',
                seasonal_periods=12
            )
            exp_smooth_fitted = exp_smooth_model.fit()
            
            exp_smooth_pred = exp_smooth_fitted.forecast(steps=len(test_data))
            exp_smooth_score = r2_score(test_data[value_col], exp_smooth_pred)
            model_scores['exp_smoothing'] = exp_smooth_score
            self.models['exp_smoothing'] = exp_smooth_fitted
            
        except Exception as e:
            logger.warning(f"Exponential Smoothing failed: {e}")
        
        # Select best model
        if model_scores:
            self.best_model = max(model_scores, key=model_scores.get)
            self.is_trained = True
            logger.info(f"Best model: {self.best_model} with R²: {model_scores[self.best_model]:.4f}")
        
        return {
            'model_scores': model_scores,
            'best_model': self.best_model
        }
    
    def forecast(self, periods: int = 365) -> pd.DataFrame:
        """Generate forecast using the best model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before forecasting")
        
        if self.best_model == 'prophet':
            future = self.models['prophet'].make_future_dataframe(periods=periods)
            forecast = self.models['prophet'].predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        
        elif self.best_model == 'arima':
            forecast = self.models['arima'].forecast(steps=periods)
            dates = pd.date_range(
                start=self.data.iloc[-1, 0] + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            return pd.DataFrame({
                'ds': dates,
                'yhat': forecast,
                'yhat_lower': forecast * 0.95,
                'yhat_upper': forecast * 1.05
            })
        
        elif self.best_model == 'exp_smoothing':
            forecast = self.models['exp_smoothing'].forecast(steps=periods)
            dates = pd.date_range(
                start=self.data.iloc[-1, 0] + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            return pd.DataFrame({
                'ds': dates,
                'yhat': forecast,
                'yhat_lower': forecast * 0.95,
                'yhat_upper': forecast * 1.05
            })
        
        return pd.DataFrame()

class HealthcareAnomalyDetector:
    """Advanced anomaly detection for healthcare metrics"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
    def train(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Train multiple anomaly detection models"""
        logger.info("Training healthcare anomaly detection models...")
        
        # Prepare features
        X_processed = self._prepare_features(X)
        
        # Initialize models
        self.models = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'one_class_svm': None,  # Will implement if needed
            'local_outlier_factor': None  # Will implement if needed
        }
        
        # Scale data
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X_processed)
        
        # Train Isolation Forest
        self.models['isolation_forest'].fit(X_scaled)
        
        # Evaluate on training data
        anomaly_scores = self.models['isolation_forest'].decision_function(X_scaled)
        anomaly_labels = self.models['isolation_forest'].predict(X_scaled)
        
        self.is_trained = True
        
        return {
            'anomaly_ratio': (anomaly_labels == -1).mean(),
            'mean_anomaly_score': anomaly_scores.mean(),
            'std_anomaly_score': anomaly_scores.std()
        }
    
    def detect_anomalies(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Detect anomalies in new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")
        
        X_processed = self._prepare_features(X)
        X_scaled = self.scalers['standard'].transform(X_processed)
        
        # Get anomaly scores and labels
        anomaly_scores = self.models['isolation_forest'].decision_function(X_scaled)
        anomaly_labels = self.models['isolation_forest'].predict(X_scaled)
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomaly_labels': anomaly_labels,
            'is_anomaly': anomaly_labels == -1
        }
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection"""
        features = df.copy()
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
        
        # Fill missing values
        features = features.fillna(features.median())
        
        return features.select_dtypes(include=[np.number]).values

class AdvancedModelManager:
    """Enhanced model manager with advanced ML capabilities"""
    
    def __init__(self):
        self.models = {
            'deep_cost_predictor': DeepHealthcareCostPredictor(),
            'ensemble_predictor': EnsembleHealthcarePredictor(),
            'advanced_forecaster': AdvancedTimeSeriesForecaster(),
            'anomaly_detector': HealthcareAnomalyDetector()
        }
        self.model_cache_dir = settings.MODEL_CACHE_DIR
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
    def train_all_models(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all advanced models"""
        results = {}
        
        # Train deep cost predictor
        if 'cost_features' in data and 'cost_targets' in data:
            try:
                results['deep_cost_predictor'] = self.models['deep_cost_predictor'].train(
                    data['cost_features'], data['cost_targets']
                )
            except Exception as e:
                logger.error(f"Deep cost predictor training failed: {e}")
        
        # Train ensemble predictor
        if 'ensemble_features' in data and 'ensemble_targets' in data:
            try:
                results['ensemble_predictor'] = self.models['ensemble_predictor'].train(
                    data['ensemble_features'], data['ensemble_targets']
                )
            except Exception as e:
                logger.error(f"Ensemble predictor training failed: {e}")
        
        # Train time series forecaster
        if 'time_series_data' in data:
            try:
                results['advanced_forecaster'] = self.models['advanced_forecaster'].train(
                    data['time_series_data'], 'date', 'value'
                )
            except Exception as e:
                logger.error(f"Advanced forecaster training failed: {e}")
        
        # Train anomaly detector
        if 'anomaly_features' in data:
            try:
                results['anomaly_detector'] = self.models['anomaly_detector'].train(
                    data['anomaly_features']
                )
            except Exception as e:
                logger.error(f"Anomaly detector training failed: {e}")
        
        return results
    
    def get_comprehensive_predictions(self, input_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get predictions from all trained models"""
        predictions = {}
        
        # Deep cost predictions
        if 'cost_features' in input_data and self.models['deep_cost_predictor'].is_trained:
            try:
                predictions['deep_cost_predictions'] = self.models['deep_cost_predictor'].predict(
                    input_data['cost_features']
                )
            except Exception as e:
                logger.error(f"Deep cost prediction failed: {e}")
        
        # Ensemble predictions
        if 'ensemble_features' in input_data and self.models['ensemble_predictor'].is_trained:
            try:
                predictions['ensemble_predictions'] = self.models['ensemble_predictor'].predict(
                    input_data['ensemble_features']
                )
            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
        
        # Time series forecast
        if self.models['advanced_forecaster'].is_trained:
            try:
                predictions['advanced_forecast'] = self.models['advanced_forecaster'].forecast(365)
            except Exception as e:
                logger.error(f"Advanced forecasting failed: {e}")
        
        # Anomaly detection
        if 'anomaly_features' in input_data and self.models['anomaly_detector'].is_trained:
            try:
                predictions['anomaly_detection'] = self.models['anomaly_detector'].detect_anomalies(
                    input_data['anomaly_features']
                )
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
        
        return predictions
    
    def save_models(self):
        """Save all trained models"""
        for name, model in self.models.items():
            if hasattr(model, 'is_trained') and model.is_trained:
                model_path = os.path.join(self.model_cache_dir, f'{name}.pkl')
                try:
                    joblib.dump(model, model_path)
                    logger.info(f"Saved {name} to {model_path}")
                except Exception as e:
                    logger.error(f"Failed to save {name}: {e}")
    
    def load_models(self):
        """Load previously trained models"""
        for name in self.models.keys():
            model_path = os.path.join(self.model_cache_dir, f'{name}.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Loaded {name} from {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load {name}: {e}")