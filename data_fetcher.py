import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Optional

class MedicalDataFetcher:
    """Fetches and processes medical industry data from various public APIs and sources."""
    
    def __init__(self):
        self.base_urls = {
            'cms': 'https://data.cms.gov/api/1/datastore/query/',
            'bls': 'https://api.bls.gov/publicAPI/v2/timeseries/data/',
            'census': 'https://api.census.gov/data/',
            'cdc': 'https://data.cdc.gov/api/views/'
        }
        
        # Cache for API responses
        self.cache = {}
        
    def get_key_metrics(self) -> Dict[str, float]:
        """Get key healthcare metrics for dashboard cards."""
        # These are realistic estimates based on public data
        return {
            'total_spending': 4300,  # $4.3 trillion in 2021
            'total_facilities': 6090,  # Approximate number of hospitals
            'total_workers': 22.0,  # 22 million healthcare workers
            'cost_per_capita': 12914  # Per capita healthcare spending
        }
    
    def get_spending_trends(self) -> pd.DataFrame:
        """Get healthcare spending trends over time."""
        # Simulated data based on CMS National Health Expenditure data
        years = list(range(2010, 2024))
        spending = [
            2594, 2701, 2817, 2963, 3123, 3337, 3492, 3649, 3808, 3978, 
            4143, 4255, 4108, 4300  # 2023 is projected
        ]
        
        return pd.DataFrame({
            'year': years,
            'spending': spending
        })
    
    def get_spending_by_category(self) -> pd.DataFrame:
        """Get healthcare spending breakdown by category."""
        categories = [
            'Hospital Care', 'Physician Services', 'Prescription Drugs',
            'Nursing Home Care', 'Dental Services', 'Home Health Care',
            'Medical Equipment', 'Other'
        ]
        amounts = [1355, 829, 378, 175, 136, 131, 61, 235]  # Billions USD
        
        return pd.DataFrame({
            'category': categories,
            'amount': amounts
        })
    
    def get_employment_by_state(self) -> pd.DataFrame:
        """Get healthcare employment data by state."""
        # Sample data for major states (in thousands)
        states_data = {
            'CA': {'employment': 1850, 'state_code': 'CA'},
            'TX': {'employment': 1420, 'state_code': 'TX'},
            'FL': {'employment': 1180, 'state_code': 'FL'},
            'NY': {'employment': 1150, 'state_code': 'NY'},
            'PA': {'employment': 780, 'state_code': 'PA'},
            'OH': {'employment': 650, 'state_code': 'OH'},
            'IL': {'employment': 620, 'state_code': 'IL'},
            'MI': {'employment': 520, 'state_code': 'MI'},
            'NC': {'employment': 510, 'state_code': 'NC'},
            'GA': {'employment': 480, 'state_code': 'GA'},
            'NJ': {'employment': 450, 'state_code': 'NJ'},
            'VA': {'employment': 420, 'state_code': 'VA'},
            'WA': {'employment': 380, 'state_code': 'WA'},
            'AZ': {'employment': 350, 'state_code': 'AZ'},
            'MA': {'employment': 340, 'state_code': 'MA'},
            'TN': {'employment': 320, 'state_code': 'TN'},
            'IN': {'employment': 310, 'state_code': 'IN'},
            'MO': {'employment': 300, 'state_code': 'MO'},
            'MD': {'employment': 290, 'state_code': 'MD'},
            'WI': {'employment': 280, 'state_code': 'WI'}
        }
        
        df_data = []
        for state, data in states_data.items():
            df_data.append({
                'state': state,
                'state_code': data['state_code'],
                'employment': data['employment']
            })
        
        return pd.DataFrame(df_data)
    
    def get_top_companies(self) -> pd.DataFrame:
        """Get top healthcare companies by revenue."""
        companies = [
            'UnitedHealth Group', 'CVS Health', 'Anthem', 'Centene',
            'HCA Healthcare', 'Kaiser Permanente', 'Humana',
            'Cigna', 'CommonSpirit Health', 'Ascension'
        ]
        revenues = [324, 268, 138, 126, 58, 95, 83, 174, 33, 27]  # Billions USD
        
        df = pd.DataFrame({
            'company': companies,
            'revenue': revenues
        })
        
        return df.sort_values('revenue', ascending=True).tail(10)
    
    def get_hospital_beds_data(self) -> pd.DataFrame:
        """Get hospital beds per 1000 population by state."""
        states = ['DC', 'ND', 'SD', 'WY', 'MT', 'MS', 'WV', 'AL', 'KS', 'NE',
                 'LA', 'OK', 'AR', 'IA', 'MO', 'KY', 'TN', 'IN', 'OH', 'PA']
        beds_per_1000 = [5.8, 5.2, 4.9, 4.7, 4.5, 4.4, 4.3, 4.1, 4.0, 3.9,
                         3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9]
        
        return pd.DataFrame({
            'state': states,
            'beds_per_1000': beds_per_1000
        })
    
    def fetch_cms_data(self, dataset_id: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch data from CMS API."""
        try:
            url = f"{self.base_urls['cms']}{dataset_id}?limit={limit}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if 'results' in data:
                return pd.DataFrame(data['results'])
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Error fetching CMS data: {e}")
            return None
    
    def fetch_bls_data(self, series_id: str, start_year: int = 2020) -> Optional[pd.DataFrame]:
        """Fetch data from Bureau of Labor Statistics API."""
        try:
            data = {
                'seriesid': [series_id],
                'startyear': str(start_year),
                'endyear': str(datetime.now().year)
            }
            
            response = requests.post(self.base_urls['bls'], 
                                   data=json.dumps(data),
                                   headers={'Content-Type': 'application/json'},
                                   timeout=10)
            response.raise_for_status()
            
            result = response.json()
            if 'Results' in result and 'series' in result['Results']:
                series_data = result['Results']['series'][0]['data']
                return pd.DataFrame(series_data)
            
            return None
            
        except Exception as e:
            print(f"Error fetching BLS data: {e}")
            return None
    
    def get_healthcare_price_index(self) -> pd.DataFrame:
        """Get healthcare price index data."""
        # Medical care CPI data (simulated based on BLS data)
        years = list(range(2015, 2024))
        cpi_values = [446.8, 460.0, 472.1, 484.8, 498.1, 505.6, 515.8, 540.1, 558.2]
        
        return pd.DataFrame({
            'year': years,
            'cpi': cpi_values,
            'annual_change': [0] + [round((cpi_values[i] - cpi_values[i-1]) / cpi_values[i-1] * 100, 1) 
                                  for i in range(1, len(cpi_values))]
        })
    
    def get_medicare_enrollment(self) -> pd.DataFrame:
        """Get Medicare enrollment trends."""
        years = list(range(2015, 2024))
        enrollment = [55.3, 56.8, 58.4, 59.9, 61.2, 62.6, 64.0, 65.0, 66.2]  # Millions
        
        return pd.DataFrame({
            'year': years,
            'enrollment_millions': enrollment
        })
    
    def get_physician_data(self) -> Dict[str, any]:
        """Get physician workforce data."""
        return {
            'total_physicians': 1043000,
            'primary_care': 240000,
            'specialists': 803000,
            'physicians_per_100k': 315,
            'shortage_areas': 7259  # Health Professional Shortage Areas
        }
    
    def get_hospital_quality_metrics(self) -> pd.DataFrame:
        """Get hospital quality and safety metrics."""
        metrics = [
            'Patient Safety Indicator', 'Readmission Rate', 'Mortality Rate',
            'Patient Experience', 'Effectiveness of Care', 'Timeliness of Care'
        ]
        scores = [3.2, 15.3, 2.1, 71.2, 87.4, 68.9]  # Various scales
        
        return pd.DataFrame({
            'metric': metrics,
            'score': scores
        })