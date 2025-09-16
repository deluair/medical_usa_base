#!/usr/bin/env python3
"""
Data Download Script for Medical Analytics Platform
Downloads and caches all required data from various APIs
"""

import sys
import os
import logging
from datetime import datetime
import pandas as pd

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from medical_analytics.data.advanced_data_fetcher import AdvancedMedicalDataFetcher
from medical_analytics.data.data_fetcher import MedicalDataFetcher
from config.settings import settings, DATA_SOURCES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Downloads and caches all required data sources"""
    
    def __init__(self):
        self.basic_fetcher = MedicalDataFetcher()
        self.advanced_fetcher = AdvancedMedicalDataFetcher()
        self.download_stats = {
            'total_datasets': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'start_time': datetime.now(),
            'datasets': {}
        }
    
    def download_cms_data(self):
        """Download CMS datasets"""
        logger.info("Downloading CMS data...")
        
        cms_datasets = [
            {'id': '9ukw-x9uj', 'name': 'Medicare Provider Utilization', 'limit': 10000},
            {'id': 'xubh-q36u', 'name': 'Hospital Compare', 'limit': 5000},
            {'id': 'hadm-7sug', 'name': 'Open Payments', 'limit': 50000},
        ]
        
        for dataset in cms_datasets:
            try:
                logger.info(f"Downloading {dataset['name']} (ID: {dataset['id']})...")
                df = self.advanced_fetcher.fetch_cms_data(dataset['id'], limit=dataset['limit'])
                
                if df is not None and not df.empty:
                    self.download_stats['successful_downloads'] += 1
                    self.download_stats['datasets'][dataset['name']] = {
                        'status': 'success',
                        'records': len(df),
                        'columns': list(df.columns) if hasattr(df, 'columns') else []
                    }
                    logger.info(f"Successfully downloaded {len(df)} records for {dataset['name']}")
                else:
                    self.download_stats['failed_downloads'] += 1
                    self.download_stats['datasets'][dataset['name']] = {
                        'status': 'failed',
                        'error': 'No data returned'
                    }
                    logger.warning(f"No data returned for {dataset['name']}")
                    
            except Exception as e:
                self.download_stats['failed_downloads'] += 1
                self.download_stats['datasets'][dataset['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Error downloading {dataset['name']}: {e}")
            
            self.download_stats['total_datasets'] += 1
    
    def download_bls_data(self):
        """Download BLS employment data"""
        logger.info("Downloading BLS employment data...")
        
        try:
            df = self.advanced_fetcher.fetch_bls_healthcare_employment()
            
            if df is not None and not df.empty:
                self.download_stats['successful_downloads'] += 1
                self.download_stats['datasets']['BLS Healthcare Employment'] = {
                    'status': 'success',
                    'records': len(df),
                    'columns': list(df.columns) if hasattr(df, 'columns') else []
                }
                logger.info(f"Successfully downloaded {len(df)} BLS employment records")
            else:
                self.download_stats['failed_downloads'] += 1
                self.download_stats['datasets']['BLS Healthcare Employment'] = {
                    'status': 'failed',
                    'error': 'No data returned'
                }
                logger.warning("No BLS employment data returned")
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            self.download_stats['datasets']['BLS Healthcare Employment'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Error downloading BLS data: {e}")
        
        self.download_stats['total_datasets'] += 1
    
    def download_fred_data(self):
        """Download FRED economic data"""
        logger.info("Downloading FRED economic data...")
        
        try:
            fred_data = self.advanced_fetcher.fred_client.get_healthcare_economic_indicators()
            
            if fred_data:
                self.download_stats['successful_downloads'] += 1
                self.download_stats['datasets']['FRED Economic Indicators'] = {
                    'status': 'success',
                    'records': len(fred_data),
                    'indicators': list(fred_data.keys()) if isinstance(fred_data, dict) else []
                }
                logger.info(f"Successfully downloaded {len(fred_data)} FRED indicators")
            else:
                self.download_stats['failed_downloads'] += 1
                self.download_stats['datasets']['FRED Economic Indicators'] = {
                    'status': 'failed',
                    'error': 'No data returned'
                }
                logger.warning("No FRED data returned")
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            self.download_stats['datasets']['FRED Economic Indicators'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Error downloading FRED data: {e}")
        
        self.download_stats['total_datasets'] += 1
    
    def download_census_data(self):
        """Download Census demographic data"""
        logger.info("Downloading Census demographic data...")
        
        try:
            census_data = self.advanced_fetcher.census_client.get_comprehensive_demographics()
            
            if census_data:
                self.download_stats['successful_downloads'] += 1
                self.download_stats['datasets']['Census Demographics'] = {
                    'status': 'success',
                    'datasets': len(census_data),
                    'categories': list(census_data.keys()) if isinstance(census_data, dict) else []
                }
                logger.info(f"Successfully downloaded {len(census_data)} Census datasets")
            else:
                self.download_stats['failed_downloads'] += 1
                self.download_stats['datasets']['Census Demographics'] = {
                    'status': 'failed',
                    'error': 'No data returned'
                }
                logger.warning("No Census data returned")
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            self.download_stats['datasets']['Census Demographics'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Error downloading Census data: {e}")
        
        self.download_stats['total_datasets'] += 1
    
    def download_comprehensive_data(self):
        """Download comprehensive data using the advanced fetcher"""
        logger.info("Downloading comprehensive medical data...")
        
        try:
            comprehensive_data = self.advanced_fetcher.get_comprehensive_data()
            
            if comprehensive_data:
                self.download_stats['successful_downloads'] += 1
                self.download_stats['datasets']['Comprehensive Medical Data'] = {
                    'status': 'success',
                    'components': list(comprehensive_data.keys()),
                    'last_updated': comprehensive_data.get('last_updated', 'Unknown')
                }
                logger.info("Successfully downloaded comprehensive medical data")
                
                # Log details about each component
                for key, value in comprehensive_data.items():
                    if isinstance(value, pd.DataFrame):
                        logger.info(f"  {key}: {len(value)} records")
                    elif isinstance(value, dict):
                        logger.info(f"  {key}: {len(value)} items")
                    else:
                        logger.info(f"  {key}: {type(value)}")
            else:
                self.download_stats['failed_downloads'] += 1
                self.download_stats['datasets']['Comprehensive Medical Data'] = {
                    'status': 'failed',
                    'error': 'No data returned'
                }
                logger.warning("No comprehensive data returned")
                
        except Exception as e:
            self.download_stats['failed_downloads'] += 1
            self.download_stats['datasets']['Comprehensive Medical Data'] = {
                'status': 'failed',
                'error': str(e)
            }
            logger.error(f"Error downloading comprehensive data: {e}")
        
        self.download_stats['total_datasets'] += 1
    
    def download_all_data(self):
        """Download all available data sources"""
        logger.info("Starting comprehensive data download...")
        logger.info(f"Cache database path: {settings.CACHE_DB_PATH}")
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(settings.CACHE_DB_PATH), exist_ok=True)
        
        # Download individual datasets
        self.download_cms_data()
        self.download_bls_data()
        self.download_fred_data()
        self.download_census_data()
        
        # Download comprehensive data (this will cache everything)
        self.download_comprehensive_data()
        
        # Calculate final stats
        self.download_stats['end_time'] = datetime.now()
        self.download_stats['duration'] = (
            self.download_stats['end_time'] - self.download_stats['start_time']
        ).total_seconds()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print download summary"""
        logger.info("=" * 60)
        logger.info("DATA DOWNLOAD SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total datasets attempted: {self.download_stats['total_datasets']}")
        logger.info(f"Successful downloads: {self.download_stats['successful_downloads']}")
        logger.info(f"Failed downloads: {self.download_stats['failed_downloads']}")
        logger.info(f"Success rate: {(self.download_stats['successful_downloads'] / max(1, self.download_stats['total_datasets']) * 100):.1f}%")
        logger.info(f"Duration: {self.download_stats['duration']:.1f} seconds")
        logger.info("")
        
        logger.info("Dataset Details:")
        for name, details in self.download_stats['datasets'].items():
            status = details['status']
            if status == 'success':
                records = details.get('records', details.get('datasets', 'N/A'))
                logger.info(f"  ✓ {name}: {records} records")
            else:
                error = details.get('error', 'Unknown error')
                logger.info(f"  ✗ {name}: {error}")
        
        logger.info("=" * 60)
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.advanced_fetcher.close()
        except:
            pass

def main():
    """Main function"""
    logger.info("Medical Analytics Platform - Data Download Script")
    logger.info("=" * 60)
    
    downloader = DataDownloader()
    
    try:
        downloader.download_all_data()
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
    finally:
        downloader.cleanup()
        logger.info("Data download script completed")

if __name__ == "__main__":
    main()