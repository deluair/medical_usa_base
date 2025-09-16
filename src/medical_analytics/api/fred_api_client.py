"""
FRED API Client for Economic Healthcare Indicators
Fetches macroeconomic data related to healthcare from the Federal Reserve Economic Data (FRED) API
"""

import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import logging
from functools import lru_cache
import time

from config import settings

logger = logging.getLogger(__name__)

class FREDAPIClient:
    """Client for fetching economic data from FRED API"""
    
    def __init__(self):
        self.api_key = settings.FRED_API_KEY
        self.base_url = settings.FRED_BASE_URL
        self.session = requests.Session()
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.last_request_time = 0
        
        # Healthcare-related economic indicators
        self.healthcare_series = {
            # Healthcare spending and costs
            'healthcare_pce': 'DHLCRG3A086NBEA',  # Healthcare Personal Consumption Expenditures
            'healthcare_price_index': 'CUUR0000SAM',  # Medical Care CPI
            'prescription_drug_prices': 'CUUR0000SEMF01',  # Prescription Drug CPI
            'hospital_services_prices': 'CUUR0000SEMF02',  # Hospital Services CPI
            'physician_services_prices': 'CUUR0000SEMF03',  # Physician Services CPI
            
            # Economic indicators affecting healthcare
            'gdp': 'GDP',  # Gross Domestic Product
            'unemployment_rate': 'UNRATE',  # Unemployment Rate
            'inflation_rate': 'CPIAUCSL',  # Consumer Price Index
            'personal_income': 'PI',  # Personal Income
            'disposable_income': 'DSPI',  # Disposable Personal Income
            
            # Healthcare employment and wages
            'healthcare_employment': 'CES6562000001',  # Healthcare Employment
            'healthcare_wages': 'CES6562000003',  # Healthcare Average Hourly Earnings
            
            # Government healthcare spending
            'medicare_spending': 'W019RCQ027SBEA',  # Medicare Government Spending
            'medicaid_spending': 'W020RCQ027SBEA',  # Medicaid Government Spending
            
            # Demographics affecting healthcare
            'population_65_over': 'LFWA64TTUSM647S',  # Population 65 and Over
            'life_expectancy': 'SPDYNLE00INUSA',  # Life Expectancy at Birth
            
            # Healthcare investment and R&D
            'healthcare_rd_spending': 'Y694RC1Q027SBEA',  # Healthcare R&D Spending
            'biotech_investment': 'NASDAQBIOTECH',  # NASDAQ Biotechnology Index
        }
    
    def _rate_limit(self):
        """Implement rate limiting for FRED API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Make a request to FRED API with error handling"""
        if not self.api_key:
            logger.warning("FRED API key not configured")
            return None
        
        self._rate_limit()
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FRED API response: {e}")
            return None
    
    def get_series_data(self, series_id: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch data for a specific FRED series"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')  # 5 years ago
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'series_id': series_id,
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'm',  # Monthly frequency
            'aggregation_method': 'avg'
        }
        
        data = self._make_request('series/observations', params)
        
        if data and 'observations' in data:
            observations = data['observations']
            df = pd.DataFrame(observations)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
                df['series_id'] = series_id
                return df[['date', 'value', 'series_id']].sort_values('date')
        
        return None
    
    def get_healthcare_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """Fetch all healthcare-related economic indicators"""
        logger.info("Fetching healthcare economic indicators from FRED")
        
        results = {}
        
        for indicator_name, series_id in self.healthcare_series.items():
            logger.info(f"Fetching {indicator_name} ({series_id})")
            
            df = self.get_series_data(series_id)
            if df is not None and not df.empty:
                df['indicator'] = indicator_name
                results[indicator_name] = df
                logger.info(f"Successfully fetched {len(df)} observations for {indicator_name}")
            else:
                logger.warning(f"No data retrieved for {indicator_name} ({series_id})")
        
        return results
    
    def get_healthcare_spending_trends(self) -> pd.DataFrame:
        """Get healthcare spending trends from FRED"""
        spending_series = [
            'healthcare_pce',
            'medicare_spending', 
            'medicaid_spending'
        ]
        
        all_data = []
        
        for series_name in spending_series:
            if series_name in self.healthcare_series:
                series_id = self.healthcare_series[series_name]
                df = self.get_series_data(series_id)
                
                if df is not None and not df.empty:
                    df['spending_type'] = series_name
                    all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        
        return pd.DataFrame()
    
    def get_healthcare_price_indices(self) -> pd.DataFrame:
        """Get healthcare price indices from FRED"""
        price_series = [
            'healthcare_price_index',
            'prescription_drug_prices',
            'hospital_services_prices',
            'physician_services_prices'
        ]
        
        all_data = []
        
        for series_name in price_series:
            if series_name in self.healthcare_series:
                series_id = self.healthcare_series[series_name]
                df = self.get_series_data(series_id)
                
                if df is not None and not df.empty:
                    df['price_category'] = series_name
                    all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df
        
        return pd.DataFrame()
    
    def get_economic_context(self) -> Dict[str, pd.DataFrame]:
        """Get broader economic context that affects healthcare"""
        context_series = [
            'gdp',
            'unemployment_rate',
            'inflation_rate',
            'personal_income',
            'disposable_income'
        ]
        
        results = {}
        
        for series_name in context_series:
            if series_name in self.healthcare_series:
                series_id = self.healthcare_series[series_name]
                df = self.get_series_data(series_id)
                
                if df is not None and not df.empty:
                    results[series_name] = df
        
        return results
    
    @lru_cache(maxsize=32)
    def get_series_info(self, series_id: str) -> Optional[Dict]:
        """Get metadata about a FRED series"""
        params = {'series_id': series_id}
        data = self._make_request('series', params)
        
        if data and 'seriess' in data and len(data['seriess']) > 0:
            return data['seriess'][0]
        
        return None
    
    def get_healthcare_correlations(self) -> pd.DataFrame:
        """Calculate correlations between healthcare indicators and economic factors"""
        try:
            # Get key healthcare and economic indicators
            healthcare_data = self.get_healthcare_economic_indicators()
            
            if not healthcare_data:
                return pd.DataFrame()
            
            # Prepare data for correlation analysis
            correlation_data = {}
            
            for indicator, df in healthcare_data.items():
                if not df.empty:
                    # Resample to monthly frequency and get latest values
                    df_monthly = df.set_index('date').resample('M')['value'].last()
                    correlation_data[indicator] = df_monthly
            
            if correlation_data:
                # Create correlation matrix
                corr_df = pd.DataFrame(correlation_data)
                correlation_matrix = corr_df.corr()
                
                return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating healthcare correlations: {e}")
        
        return pd.DataFrame()
    
    def get_healthcare_forecasting_data(self) -> Dict[str, pd.DataFrame]:
        """Get data suitable for healthcare economic forecasting"""
        forecasting_series = [
            'healthcare_pce',
            'healthcare_price_index',
            'healthcare_employment',
            'gdp',
            'population_65_over'
        ]
        
        results = {}
        
        for series_name in forecasting_series:
            if series_name in self.healthcare_series:
                series_id = self.healthcare_series[series_name]
                # Get longer time series for forecasting (10 years)
                start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
                df = self.get_series_data(series_id, start_date=start_date)
                
                if df is not None and not df.empty:
                    results[series_name] = df
        
        return results