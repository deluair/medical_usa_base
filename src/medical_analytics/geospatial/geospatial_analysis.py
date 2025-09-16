"""
Geospatial Analysis for Healthcare Accessibility and Healthcare Deserts
Includes hospital accessibility mapping, healthcare deserts identification, and spatial analytics
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopy.distance import geodesic
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config.settings import settings

logger = logging.getLogger(__name__)

class HealthcareAccessibilityAnalyzer:
    """Analyzes healthcare accessibility and identifies healthcare deserts"""
    
    def __init__(self):
        self.hospitals_gdf = None
        self.population_gdf = None
        self.accessibility_scores = None
        
    def load_hospital_data(self, hospitals_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert hospital data to GeoDataFrame"""
        if 'latitude' in hospitals_df.columns and 'longitude' in hospitals_df.columns:
            geometry = [Point(xy) for xy in zip(hospitals_df.longitude, hospitals_df.latitude)]
            self.hospitals_gdf = gpd.GeoDataFrame(hospitals_df, geometry=geometry, crs='EPSG:4326')
            logger.info(f"Loaded {len(self.hospitals_gdf)} hospitals for geospatial analysis")
            return self.hospitals_gdf
        else:
            logger.error("Hospital data must contain 'latitude' and 'longitude' columns")
            return None
    
    def load_population_data(self, population_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Load population data with geographic information"""
        if 'latitude' in population_df.columns and 'longitude' in population_df.columns:
            geometry = [Point(xy) for xy in zip(population_df.longitude, population_df.latitude)]
            self.population_gdf = gpd.GeoDataFrame(population_df, geometry=geometry, crs='EPSG:4326')
            return self.population_gdf
        else:
            # Create synthetic population data if not available
            return self._create_synthetic_population_data()
    
    def _create_synthetic_population_data(self) -> gpd.GeoDataFrame:
        """Create synthetic population data for analysis"""
        logger.info("Creating synthetic population data for accessibility analysis")
        
        # Generate population centers for major US cities
        major_cities = [
            {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'population': 8336817},
            {'city': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437, 'population': 3979576},
            {'city': 'Chicago', 'lat': 41.8781, 'lon': -87.6298, 'population': 2693976},
            {'city': 'Houston', 'lat': 29.7604, 'lon': -95.3698, 'population': 2320268},
            {'city': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740, 'population': 1680992},
            {'city': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652, 'population': 1584064},
            {'city': 'San Antonio', 'lat': 29.4241, 'lon': -98.4936, 'population': 1547253},
            {'city': 'San Diego', 'lat': 32.7157, 'lon': -117.1611, 'population': 1423851},
            {'city': 'Dallas', 'lat': 32.7767, 'lon': -96.7970, 'population': 1343573},
            {'city': 'San Jose', 'lat': 37.3382, 'lon': -121.8863, 'population': 1021795}
        ]
        
        # Add rural areas with lower population density
        rural_areas = []
        for i in range(50):  # Add 50 rural population centers
            lat = np.random.uniform(25, 49)  # Continental US latitude range
            lon = np.random.uniform(-125, -66)  # Continental US longitude range
            population = np.random.uniform(1000, 50000)  # Rural population range
            
            rural_areas.append({
                'city': f'Rural_Area_{i+1}',
                'lat': lat,
                'lon': lon,
                'population': population
            })
        
        all_areas = major_cities + rural_areas
        population_df = pd.DataFrame(all_areas)
        
        # Add demographic and health indicators
        population_df['elderly_population'] = population_df['population'] * np.random.uniform(0.12, 0.25, len(population_df))
        population_df['poverty_rate'] = np.random.uniform(0.05, 0.30, len(population_df))
        population_df['uninsured_rate'] = np.random.uniform(0.03, 0.20, len(population_df))
        population_df['chronic_disease_rate'] = np.random.uniform(0.15, 0.40, len(population_df))
        
        # Create geometry
        geometry = [Point(xy) for xy in zip(population_df.lon, population_df.lat)]
        self.population_gdf = gpd.GeoDataFrame(population_df, geometry=geometry, crs='EPSG:4326')
        
        return self.population_gdf
    
    def calculate_accessibility_scores(self, max_distance_km: float = 50) -> pd.DataFrame:
        """Calculate healthcare accessibility scores for each population center"""
        if self.hospitals_gdf is None or self.population_gdf is None:
            logger.error("Hospital and population data must be loaded first")
            return None
        
        logger.info("Calculating healthcare accessibility scores...")
        
        accessibility_data = []
        
        for idx, pop_center in self.population_gdf.iterrows():
            pop_point = (pop_center.geometry.y, pop_center.geometry.x)
            
            # Find hospitals within max distance
            nearby_hospitals = []
            hospital_distances = []
            
            for _, hospital in self.hospitals_gdf.iterrows():
                hospital_point = (hospital.geometry.y, hospital.geometry.x)
                distance = geodesic(pop_point, hospital_point).kilometers
                
                if distance <= max_distance_km:
                    nearby_hospitals.append(hospital)
                    hospital_distances.append(distance)
            
            # Calculate accessibility metrics
            num_hospitals = len(nearby_hospitals)
            avg_distance = np.mean(hospital_distances) if hospital_distances else max_distance_km
            min_distance = min(hospital_distances) if hospital_distances else max_distance_km
            
            # Calculate hospital capacity within reach
            total_beds = sum([h.get('beds', 100) for h in nearby_hospitals]) if nearby_hospitals else 0
            beds_per_1000 = (total_beds / pop_center['population']) * 1000 if pop_center['population'] > 0 else 0
            
            # Calculate accessibility score (0-100)
            distance_score = max(0, 100 - (min_distance / max_distance_km) * 100)
            capacity_score = min(100, beds_per_1000 * 20)  # Normalize beds per 1000
            availability_score = min(100, num_hospitals * 10)  # Number of hospitals
            
            accessibility_score = (distance_score * 0.4 + capacity_score * 0.3 + availability_score * 0.3)
            
            # Adjust for demographic factors
            demographic_adjustment = 1.0
            if pop_center.get('poverty_rate', 0) > 0.2:
                demographic_adjustment -= 0.1
            if pop_center.get('elderly_population', 0) / pop_center['population'] > 0.2:
                demographic_adjustment -= 0.1
            if pop_center.get('uninsured_rate', 0) > 0.15:
                demographic_adjustment -= 0.1
            
            final_score = accessibility_score * max(0.5, demographic_adjustment)
            
            accessibility_data.append({
                'location': pop_center['city'],
                'latitude': pop_center.geometry.y,
                'longitude': pop_center.geometry.x,
                'population': pop_center['population'],
                'num_hospitals_nearby': num_hospitals,
                'min_distance_km': min_distance,
                'avg_distance_km': avg_distance,
                'total_beds_nearby': total_beds,
                'beds_per_1000': beds_per_1000,
                'accessibility_score': final_score,
                'is_healthcare_desert': final_score < 30,  # Threshold for healthcare desert
                'poverty_rate': pop_center.get('poverty_rate', 0),
                'elderly_population_pct': pop_center.get('elderly_population', 0) / pop_center['population'] * 100,
                'uninsured_rate': pop_center.get('uninsured_rate', 0)
            })
        
        self.accessibility_scores = pd.DataFrame(accessibility_data)
        logger.info(f"Calculated accessibility scores for {len(self.accessibility_scores)} locations")
        
        return self.accessibility_scores
    
    def identify_healthcare_deserts(self, threshold_score: float = 30) -> pd.DataFrame:
        """Identify healthcare deserts based on accessibility scores"""
        if self.accessibility_scores is None:
            logger.error("Accessibility scores must be calculated first")
            return None
        
        healthcare_deserts = self.accessibility_scores[
            self.accessibility_scores['accessibility_score'] < threshold_score
        ].copy()
        
        # Calculate severity levels
        healthcare_deserts['severity'] = pd.cut(
            healthcare_deserts['accessibility_score'],
            bins=[0, 10, 20, 30],
            labels=['Critical', 'Severe', 'Moderate']
        )
        
        logger.info(f"Identified {len(healthcare_deserts)} healthcare deserts")
        
        return healthcare_deserts.sort_values('accessibility_score')
    
    def create_accessibility_map(self) -> folium.Map:
        """Create interactive map showing healthcare accessibility"""
        if self.accessibility_scores is None:
            logger.error("Accessibility scores must be calculated first")
            return None
        
        # Center map on US
        center_lat = self.accessibility_scores['latitude'].mean()
        center_lon = self.accessibility_scores['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=5,
            tiles='OpenStreetMap'
        )
        
        # Add hospitals
        if self.hospitals_gdf is not None:
            for _, hospital in self.hospitals_gdf.iterrows():
                folium.Marker(
                    location=[hospital.geometry.y, hospital.geometry.x],
                    popup=f"Hospital: {hospital.get('name', 'Unknown')}<br>Beds: {hospital.get('beds', 'N/A')}",
                    icon=folium.Icon(color='red', icon='plus', prefix='fa')
                ).add_to(m)
        
        # Add population centers with accessibility scores
        for _, location in self.accessibility_scores.iterrows():
            # Color based on accessibility score
            if location['accessibility_score'] >= 70:
                color = 'green'
            elif location['accessibility_score'] >= 40:
                color = 'orange'
            else:
                color = 'red'
            
            # Size based on population
            radius = max(5, min(20, location['population'] / 100000))
            
            folium.CircleMarker(
                location=[location['latitude'], location['longitude']],
                radius=radius,
                popup=f"""
                <b>{location['location']}</b><br>
                Population: {location['population']:,}<br>
                Accessibility Score: {location['accessibility_score']:.1f}<br>
                Nearest Hospital: {location['min_distance_km']:.1f} km<br>
                Hospitals Nearby: {location['num_hospitals_nearby']}<br>
                Healthcare Desert: {'Yes' if location['is_healthcare_desert'] else 'No'}
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Healthcare Accessibility</b></p>
        <p><i class="fa fa-circle" style="color:green"></i> High (70+)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Medium (40-69)</p>
        <p><i class="fa fa-circle" style="color:red"></i> Low (<40)</p>
        <p><i class="fa fa-plus" style="color:red"></i> Hospitals</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def analyze_spatial_patterns(self) -> Dict[str, any]:
        """Analyze spatial patterns in healthcare accessibility"""
        if self.accessibility_scores is None:
            return {}
        
        analysis = {
            'summary_stats': {
                'total_locations': len(self.accessibility_scores),
                'healthcare_deserts': len(self.accessibility_scores[self.accessibility_scores['is_healthcare_desert']]),
                'avg_accessibility_score': self.accessibility_scores['accessibility_score'].mean(),
                'median_distance_to_hospital': self.accessibility_scores['min_distance_km'].median(),
                'total_population_in_deserts': self.accessibility_scores[
                    self.accessibility_scores['is_healthcare_desert']
                ]['population'].sum()
            },
            'regional_analysis': {},
            'correlation_analysis': {}
        }
        
        # Regional analysis (simplified by latitude bands)
        self.accessibility_scores['region'] = pd.cut(
            self.accessibility_scores['latitude'],
            bins=[25, 35, 40, 45, 50],
            labels=['South', 'Southwest', 'Midwest', 'North']
        )
        
        regional_stats = self.accessibility_scores.groupby('region').agg({
            'accessibility_score': ['mean', 'std'],
            'is_healthcare_desert': 'sum',
            'population': 'sum',
            'min_distance_km': 'mean'
        }).round(2)
        
        analysis['regional_analysis'] = regional_stats.to_dict()
        
        # Correlation analysis
        numeric_cols = ['accessibility_score', 'population', 'poverty_rate', 
                       'elderly_population_pct', 'uninsured_rate', 'min_distance_km']
        
        correlation_matrix = self.accessibility_scores[numeric_cols].corr()
        analysis['correlation_analysis'] = correlation_matrix.to_dict()
        
        return analysis
    
    def generate_accessibility_report(self) -> Dict[str, any]:
        """Generate comprehensive accessibility report"""
        if self.accessibility_scores is None:
            logger.error("Accessibility scores must be calculated first")
            return {}
        
        healthcare_deserts = self.identify_healthcare_deserts()
        spatial_analysis = self.analyze_spatial_patterns()
        
        report = {
            'executive_summary': {
                'total_population_analyzed': self.accessibility_scores['population'].sum(),
                'healthcare_deserts_identified': len(healthcare_deserts),
                'population_in_deserts': healthcare_deserts['population'].sum(),
                'avg_accessibility_score': self.accessibility_scores['accessibility_score'].mean(),
                'critical_areas': len(healthcare_deserts[healthcare_deserts['severity'] == 'Critical'])
            },
            'healthcare_deserts': healthcare_deserts.to_dict('records') if healthcare_deserts is not None else [],
            'spatial_analysis': spatial_analysis,
            'recommendations': self._generate_recommendations(healthcare_deserts),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, healthcare_deserts: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis"""
        if healthcare_deserts is None or len(healthcare_deserts) == 0:
            return ["Healthcare accessibility appears adequate across analyzed regions."]
        
        recommendations = []
        
        # Critical areas
        critical_areas = healthcare_deserts[healthcare_deserts['severity'] == 'Critical']
        if len(critical_areas) > 0:
            recommendations.append(
                f"Immediate intervention needed in {len(critical_areas)} critical healthcare desert areas "
                f"affecting {critical_areas['population'].sum():,.0f} people."
            )
        
        # High poverty areas
        high_poverty_deserts = healthcare_deserts[healthcare_deserts['poverty_rate'] > 0.2]
        if len(high_poverty_deserts) > 0:
            recommendations.append(
                "Consider mobile health clinics and telemedicine programs for high-poverty healthcare deserts."
            )
        
        # Elderly population
        elderly_deserts = healthcare_deserts[healthcare_deserts['elderly_population_pct'] > 20]
        if len(elderly_deserts) > 0:
            recommendations.append(
                "Prioritize geriatric care facilities in areas with high elderly populations lacking healthcare access."
            )
        
        # Distance-based recommendations
        remote_areas = healthcare_deserts[healthcare_deserts['min_distance_km'] > 30]
        if len(remote_areas) > 0:
            recommendations.append(
                "Establish satellite clinics or improve transportation services for remote areas "
                f"where nearest hospital is >30km away."
            )
        
        return recommendations

class HealthcareGISAnalyzer:
    """Advanced GIS analysis for healthcare planning"""
    
    def __init__(self):
        self.service_areas = None
        self.catchment_areas = None
        
    def calculate_service_areas(self, hospitals_gdf: gpd.GeoDataFrame, 
                              service_radius_km: float = 25) -> gpd.GeoDataFrame:
        """Calculate service areas (catchment areas) for hospitals"""
        logger.info(f"Calculating service areas with {service_radius_km}km radius...")
        
        # Convert radius to degrees (approximate)
        radius_degrees = service_radius_km / 111.0  # 1 degree ≈ 111 km
        
        service_areas = []
        
        for idx, hospital in hospitals_gdf.iterrows():
            # Create circular buffer around hospital
            center_point = hospital.geometry
            buffer_area = center_point.buffer(radius_degrees)
            
            service_areas.append({
                'hospital_id': idx,
                'hospital_name': hospital.get('name', f'Hospital_{idx}'),
                'beds': hospital.get('beds', 100),
                'geometry': buffer_area,
                'service_radius_km': service_radius_km
            })
        
        self.service_areas = gpd.GeoDataFrame(service_areas, crs='EPSG:4326')
        return self.service_areas
    
    def analyze_coverage_gaps(self, population_gdf: gpd.GeoDataFrame) -> Dict[str, any]:
        """Analyze coverage gaps in healthcare services"""
        if self.service_areas is None:
            logger.error("Service areas must be calculated first")
            return {}
        
        coverage_analysis = {
            'covered_population': 0,
            'uncovered_population': 0,
            'coverage_percentage': 0,
            'gap_areas': [],
            'overlapping_coverage': []
        }
        
        total_population = population_gdf['population'].sum()
        
        # Check which population centers are covered
        covered_pop = 0
        uncovered_areas = []
        
        for idx, pop_center in population_gdf.iterrows():
            is_covered = False
            covering_hospitals = []
            
            for _, service_area in self.service_areas.iterrows():
                if service_area.geometry.contains(pop_center.geometry):
                    is_covered = True
                    covering_hospitals.append(service_area['hospital_name'])
            
            if is_covered:
                covered_pop += pop_center['population']
                if len(covering_hospitals) > 1:
                    coverage_analysis['overlapping_coverage'].append({
                        'location': pop_center['city'],
                        'population': pop_center['population'],
                        'covering_hospitals': covering_hospitals
                    })
            else:
                uncovered_areas.append({
                    'location': pop_center['city'],
                    'population': pop_center['population'],
                    'latitude': pop_center.geometry.y,
                    'longitude': pop_center.geometry.x
                })
        
        coverage_analysis['covered_population'] = covered_pop
        coverage_analysis['uncovered_population'] = total_population - covered_pop
        coverage_analysis['coverage_percentage'] = (covered_pop / total_population) * 100
        coverage_analysis['gap_areas'] = uncovered_areas
        
        return coverage_analysis
    
    def optimize_hospital_locations(self, population_gdf: gpd.GeoDataFrame, 
                                  num_new_hospitals: int = 5) -> List[Dict[str, any]]:
        """Suggest optimal locations for new hospitals using spatial optimization"""
        logger.info(f"Optimizing locations for {num_new_hospitals} new hospitals...")
        
        # Simple optimization: find population centers with lowest accessibility
        if hasattr(self, 'accessibility_scores') and self.accessibility_scores is not None:
            # Use existing accessibility analysis
            worst_access_areas = self.accessibility_scores.nsmallest(
                num_new_hospitals * 2, 'accessibility_score'
            )
        else:
            # Use population density and distance heuristics
            worst_access_areas = population_gdf.nlargest(num_new_hospitals * 2, 'population')
        
        optimal_locations = []
        
        for i in range(min(num_new_hospitals, len(worst_access_areas))):
            area = worst_access_areas.iloc[i]
            
            optimal_locations.append({
                'rank': i + 1,
                'suggested_location': area.get('city', f'Location_{i+1}'),
                'latitude': area.geometry.y if hasattr(area, 'geometry') else area.get('latitude'),
                'longitude': area.geometry.x if hasattr(area, 'geometry') else area.get('longitude'),
                'population_served': area.get('population', 0),
                'current_accessibility_score': area.get('accessibility_score', 0),
                'justification': f"High population density with limited healthcare access",
                'estimated_impact': area.get('population', 0) * 0.8  # Estimated population that would benefit
            })
        
        return optimal_locations