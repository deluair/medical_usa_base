"""
Census API Client for Demographic and Population Health Data
Fetches demographic data from the U.S. Census Bureau API for healthcare analytics
"""

import pandas as pd
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from functools import lru_cache
import time

from config import settings

logger = logging.getLogger(__name__)

class CensusAPIClient:
    """Client for fetching demographic data from Census API"""
    
    def __init__(self):
        self.api_key = settings.CENSUS_API_KEY
        self.base_url = settings.CENSUS_BASE_URL
        self.session = requests.Session()
        self.rate_limit_delay = 0.5  # 0.5 seconds between requests
        self.last_request_time = 0
        
        # Healthcare-relevant demographic variables
        self.demographic_variables = {
            # Population by age groups (important for healthcare demand)
            'total_population': 'B01003_001E',
            'population_under_5': 'B01001_003E',
            'population_5_to_17': 'B01001_004E',
            'population_18_to_64': 'B01001_020E',
            'population_65_plus': 'B01001_025E',
            'median_age': 'B01002_001E',
            
            # Health insurance coverage
            'total_with_insurance': 'B27001_004E',
            'total_without_insurance': 'B27001_005E',
            'private_insurance': 'B27001_002E',
            'public_insurance': 'B27001_003E',
            'medicare_coverage': 'B27007_008E',
            'medicaid_coverage': 'B27007_004E',
            
            # Income and poverty (affects healthcare access)
            'median_household_income': 'B19013_001E',
            'per_capita_income': 'B19301_001E',
            'poverty_rate': 'B17001_002E',
            'total_households': 'B25001_001E',
            
            # Education (correlates with health outcomes)
            'high_school_graduate': 'B15003_017E',
            'bachelors_degree': 'B15003_022E',
            'graduate_degree': 'B15003_025E',
            
            # Employment in healthcare
            'healthcare_workers': 'C24010_054E',
            'total_employed': 'B23025_002E',
            'unemployment_rate': 'B23025_005E',
            
            # Disability status
            'total_with_disability': 'B18101_001E',
            'disability_under_65': 'B18101_004E',
            'disability_65_plus': 'B18101_023E',
            
            # Race and ethnicity (health disparities)
            'white_alone': 'B02001_002E',
            'black_alone': 'B02001_003E',
            'asian_alone': 'B02001_005E',
            'hispanic_latino': 'B03003_003E',
            
            # Housing characteristics (social determinants of health)
            'owner_occupied_housing': 'B25003_002E',
            'renter_occupied_housing': 'B25003_003E',
            'median_home_value': 'B25077_001E',
            'median_rent': 'B25064_001E',
        }
        
        # State FIPS codes for geographic analysis
        self.state_fips = {
            'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06', 'CO': '08',
            'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13', 'HI': '15', 'ID': '16',
            'IL': '17', 'IN': '18', 'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22',
            'ME': '23', 'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
            'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
            'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39', 'OK': '40',
            'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47',
            'TX': '48', 'UT': '49', 'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54',
            'WI': '55', 'WY': '56', 'DC': '11', 'PR': '72'
        }
    
    def _rate_limit(self):
        """Implement rate limiting for Census API"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[List]:
        """Make a request to Census API with error handling"""
        self._rate_limit()
        
        if self.api_key:
            params['key'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Census API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Census API response: {e}")
            return None
    
    def get_state_demographics(self, variables: List[str] = None, year: int = 2021) -> pd.DataFrame:
        """Get demographic data for all states"""
        if variables is None:
            variables = list(self.demographic_variables.keys())
        
        # Convert variable names to Census codes
        census_vars = []
        for var in variables:
            if var in self.demographic_variables:
                census_vars.append(self.demographic_variables[var])
        
        if not census_vars:
            logger.warning("No valid variables specified")
            return pd.DataFrame()
        
        # Build the API request
        get_vars = ','.join(census_vars + ['NAME'])
        params = {
            'get': get_vars,
            'for': 'state:*'
        }
        
        endpoint = f"{year}/acs/acs5"
        data = self._make_request(endpoint, params)
        
        if data and len(data) > 1:  # First row is headers
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Convert numeric columns
            for i, var in enumerate(census_vars):
                col_name = headers[i]
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            # Add readable variable names
            var_mapping = {v: k for k, v in self.demographic_variables.items() if v in headers}
            df = df.rename(columns=var_mapping)
            
            # Clean up state names and add state codes
            df['state_name'] = df['NAME']
            df['state_code'] = df['state'].map({v: k for k, v in self.state_fips.items()})
            
            return df
        
        return pd.DataFrame()
    
    def get_county_demographics(self, state_fips: str, variables: List[str] = None, year: int = 2021) -> pd.DataFrame:
        """Get demographic data for counties in a specific state"""
        if variables is None:
            variables = ['total_population', 'median_age', 'median_household_income', 
                        'total_with_insurance', 'total_without_insurance']
        
        # Convert variable names to Census codes
        census_vars = []
        for var in variables:
            if var in self.demographic_variables:
                census_vars.append(self.demographic_variables[var])
        
        if not census_vars:
            return pd.DataFrame()
        
        # Build the API request
        get_vars = ','.join(census_vars + ['NAME'])
        params = {
            'get': get_vars,
            'for': f'county:*',
            'in': f'state:{state_fips}'
        }
        
        endpoint = f"{year}/acs/acs5"
        data = self._make_request(endpoint, params)
        
        if data and len(data) > 1:
            headers = data[0]
            rows = data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            
            # Convert numeric columns
            for i, var in enumerate(census_vars):
                col_name = headers[i]
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            # Add readable variable names
            var_mapping = {v: k for k, v in self.demographic_variables.items() if v in headers}
            df = df.rename(columns=var_mapping)
            
            df['county_name'] = df['NAME']
            df['state_fips'] = state_fips
            
            return df
        
        return pd.DataFrame()
    
    def get_healthcare_access_indicators(self) -> pd.DataFrame:
        """Get indicators related to healthcare access and coverage"""
        healthcare_vars = [
            'total_population',
            'total_with_insurance',
            'total_without_insurance',
            'medicare_coverage',
            'medicaid_coverage',
            'private_insurance',
            'median_household_income',
            'poverty_rate',
            'population_65_plus',
            'total_with_disability'
        ]
        
        df = self.get_state_demographics(healthcare_vars)
        
        if not df.empty:
            # Calculate derived metrics
            df['uninsured_rate'] = (df['total_without_insurance'] / df['total_population']) * 100
            df['medicare_rate'] = (df['medicare_coverage'] / df['total_population']) * 100
            df['medicaid_rate'] = (df['medicaid_coverage'] / df['total_population']) * 100
            df['elderly_population_rate'] = (df['population_65_plus'] / df['total_population']) * 100
            df['disability_rate'] = (df['total_with_disability'] / df['total_population']) * 100
        
        return df
    
    def get_social_determinants_of_health(self) -> pd.DataFrame:
        """Get social determinants of health data"""
        sdoh_vars = [
            'total_population',
            'median_household_income',
            'per_capita_income',
            'poverty_rate',
            'high_school_graduate',
            'bachelors_degree',
            'unemployment_rate',
            'owner_occupied_housing',
            'median_home_value',
            'median_rent'
        ]
        
        df = self.get_state_demographics(sdoh_vars)
        
        if not df.empty:
            # Calculate derived metrics
            df['poverty_rate_pct'] = (df['poverty_rate'] / df['total_population']) * 100
            df['education_rate'] = ((df['high_school_graduate'] + df['bachelors_degree']) / df['total_population']) * 100
            df['homeownership_rate'] = (df['owner_occupied_housing'] / (df['owner_occupied_housing'] + df['renter_occupied_housing'])) * 100
        
        return df
    
    def get_health_disparities_data(self) -> pd.DataFrame:
        """Get data for analyzing health disparities by race/ethnicity"""
        disparity_vars = [
            'total_population',
            'white_alone',
            'black_alone',
            'asian_alone',
            'hispanic_latino',
            'total_with_insurance',
            'total_without_insurance',
            'median_household_income',
            'poverty_rate'
        ]
        
        df = self.get_state_demographics(disparity_vars)
        
        if not df.empty:
            # Calculate demographic percentages
            df['white_pct'] = (df['white_alone'] / df['total_population']) * 100
            df['black_pct'] = (df['black_alone'] / df['total_population']) * 100
            df['asian_pct'] = (df['asian_alone'] / df['total_population']) * 100
            df['hispanic_pct'] = (df['hispanic_latino'] / df['total_population']) * 100
            df['uninsured_rate'] = (df['total_without_insurance'] / df['total_population']) * 100
        
        return df
    
    def get_healthcare_workforce_data(self) -> pd.DataFrame:
        """Get data about healthcare workforce by state"""
        workforce_vars = [
            'total_population',
            'healthcare_workers',
            'total_employed',
            'unemployment_rate',
            'bachelors_degree',
            'graduate_degree'
        ]
        
        df = self.get_state_demographics(workforce_vars)
        
        if not df.empty:
            # Calculate workforce metrics
            df['healthcare_workers_per_1000'] = (df['healthcare_workers'] / df['total_population']) * 1000
            df['healthcare_employment_rate'] = (df['healthcare_workers'] / df['total_employed']) * 100
            df['advanced_education_rate'] = ((df['bachelors_degree'] + df['graduate_degree']) / df['total_population']) * 100
        
        return df
    
    @lru_cache(maxsize=50)
    def get_variable_info(self, variable_code: str, year: int = 2021) -> Optional[Dict]:
        """Get metadata about a Census variable"""
        try:
            endpoint = f"{year}/acs/acs5/variables/{variable_code}"
            params = {}
            data = self._make_request(endpoint, params)
            return data
        except Exception as e:
            logger.error(f"Error fetching variable info for {variable_code}: {e}")
            return None
    
    def get_comprehensive_demographics(self) -> Dict[str, pd.DataFrame]:
        """Get comprehensive demographic data for healthcare analysis"""
        logger.info("Fetching comprehensive demographic data from Census API")
        
        results = {}
        
        try:
            results['healthcare_access'] = self.get_healthcare_access_indicators()
            logger.info("Fetched healthcare access indicators")
        except Exception as e:
            logger.error(f"Error fetching healthcare access data: {e}")
            results['healthcare_access'] = pd.DataFrame()
        
        try:
            results['social_determinants'] = self.get_social_determinants_of_health()
            logger.info("Fetched social determinants of health")
        except Exception as e:
            logger.error(f"Error fetching social determinants data: {e}")
            results['social_determinants'] = pd.DataFrame()
        
        try:
            results['health_disparities'] = self.get_health_disparities_data()
            logger.info("Fetched health disparities data")
        except Exception as e:
            logger.error(f"Error fetching health disparities data: {e}")
            results['health_disparities'] = pd.DataFrame()
        
        try:
            results['healthcare_workforce'] = self.get_healthcare_workforce_data()
            logger.info("Fetched healthcare workforce data")
        except Exception as e:
            logger.error(f"Error fetching healthcare workforce data: {e}")
            results['healthcare_workforce'] = pd.DataFrame()
        
        return results