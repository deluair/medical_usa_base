"""
Enhanced Geospatial Analysis Module for Healthcare Analytics
Provides advanced mapping, clustering, and spatial intelligence capabilities
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import requests
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from geopy.distance import geodesic
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class HealthcareFacility:
    """Healthcare facility data structure"""
    name: str
    latitude: float
    longitude: float
    facility_type: str
    capacity: int
    services: List[str]
    quality_rating: float
    address: str

class AdvancedGeospatialAnalyzer:
    """Advanced geospatial analysis for healthcare data"""
    
    def __init__(self):
        self.facilities = []
        self.population_data = None
        self.accessibility_scores = {}
        
    def load_healthcare_facilities(self, data_source: str = "cms") -> List[HealthcareFacility]:
        """Load healthcare facilities from various data sources"""
        try:
            # Sample healthcare facilities data
            facilities_data = [
                {
                    'name': 'Mayo Clinic',
                    'latitude': 44.0225,
                    'longitude': -92.4699,
                    'facility_type': 'Hospital',
                    'capacity': 1265,
                    'services': ['Emergency', 'Surgery', 'Cardiology', 'Oncology'],
                    'quality_rating': 4.8,
                    'address': 'Rochester, MN'
                },
                {
                    'name': 'Cleveland Clinic',
                    'latitude': 41.5034,
                    'longitude': -81.6234,
                    'facility_type': 'Hospital',
                    'capacity': 1285,
                    'services': ['Emergency', 'Surgery', 'Neurology', 'Cardiology'],
                    'quality_rating': 4.7,
                    'address': 'Cleveland, OH'
                },
                {
                    'name': 'Johns Hopkins Hospital',
                    'latitude': 39.2970,
                    'longitude': -76.5929,
                    'facility_type': 'Hospital',
                    'capacity': 1154,
                    'services': ['Emergency', 'Surgery', 'Pediatrics', 'Research'],
                    'quality_rating': 4.9,
                    'address': 'Baltimore, MD'
                }
            ]
            
            self.facilities = [HealthcareFacility(**facility) for facility in facilities_data]
            logger.info(f"Loaded {len(self.facilities)} healthcare facilities")
            return self.facilities
            
        except Exception as e:
            logger.error(f"Error loading healthcare facilities: {e}")
            return []
    
    def calculate_accessibility_scores(self, population_centers: List[Tuple[float, float]]) -> Dict:
        """Calculate healthcare accessibility scores for population centers"""
        try:
            accessibility_data = []
            
            for i, (lat, lon) in enumerate(population_centers):
                # Calculate distance to nearest healthcare facility
                min_distance = float('inf')
                nearest_facility = None
                
                for facility in self.facilities:
                    distance = geodesic((lat, lon), (facility.latitude, facility.longitude)).kilometers
                    if distance < min_distance:
                        min_distance = distance
                        nearest_facility = facility
                
                # Calculate accessibility score (inverse of distance, weighted by quality)
                if nearest_facility and min_distance > 0:
                    base_score = 1 / (1 + min_distance / 10)  # Normalize distance
                    quality_weight = nearest_facility.quality_rating / 5.0
                    accessibility_score = base_score * quality_weight
                else:
                    accessibility_score = 0
                
                accessibility_data.append({
                    'population_center_id': i,
                    'latitude': lat,
                    'longitude': lon,
                    'nearest_facility': nearest_facility.name if nearest_facility else None,
                    'distance_km': min_distance if min_distance != float('inf') else None,
                    'accessibility_score': accessibility_score
                })
            
            self.accessibility_scores = {
                'data': accessibility_data,
                'summary': {
                    'avg_accessibility': np.mean([d['accessibility_score'] for d in accessibility_data]),
                    'min_accessibility': min([d['accessibility_score'] for d in accessibility_data]),
                    'max_accessibility': max([d['accessibility_score'] for d in accessibility_data])
                }
            }
            
            return self.accessibility_scores
            
        except Exception as e:
            logger.error(f"Error calculating accessibility scores: {e}")
            return {}
    
    def identify_healthcare_deserts(self, threshold_distance: float = 50.0) -> Dict:
        """Identify healthcare desert areas"""
        try:
            if not self.accessibility_scores:
                return {}
            
            desert_areas = []
            for data_point in self.accessibility_scores['data']:
                if data_point['distance_km'] and data_point['distance_km'] > threshold_distance:
                    desert_areas.append({
                        'latitude': data_point['latitude'],
                        'longitude': data_point['longitude'],
                        'distance_to_care': data_point['distance_km'],
                        'severity': 'High' if data_point['distance_km'] > 100 else 'Medium'
                    })
            
            return {
                'desert_areas': desert_areas,
                'total_desert_areas': len(desert_areas),
                'high_severity_count': len([d for d in desert_areas if d['severity'] == 'High'])
            }
            
        except Exception as e:
            logger.error(f"Error identifying healthcare deserts: {e}")
            return {}
    
    def perform_spatial_clustering(self, data_points: List[Dict]) -> Dict:
        """Perform spatial clustering analysis on healthcare data"""
        try:
            if not data_points:
                return {}
            
            # Prepare data for clustering
            coordinates = np.array([[point['latitude'], point['longitude']] for point in data_points])
            
            # Standardize coordinates
            scaler = StandardScaler()
            coordinates_scaled = scaler.fit_transform(coordinates)
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            dbscan_labels = dbscan.fit_predict(coordinates_scaled)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans_labels = kmeans.fit_predict(coordinates_scaled)
            
            # Add cluster labels to data points
            clustered_data = []
            for i, point in enumerate(data_points):
                clustered_point = point.copy()
                clustered_point['dbscan_cluster'] = int(dbscan_labels[i])
                clustered_point['kmeans_cluster'] = int(kmeans_labels[i])
                clustered_data.append(clustered_point)
            
            return {
                'clustered_data': clustered_data,
                'dbscan_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'kmeans_clusters': len(set(kmeans_labels)),
                'noise_points': list(dbscan_labels).count(-1)
            }
            
        except Exception as e:
            logger.error(f"Error in spatial clustering: {e}")
            return {}
    
    def create_interactive_map(self, center_lat: float = 39.8283, center_lon: float = -98.5795) -> folium.Map:
        """Create interactive healthcare accessibility map"""
        try:
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=4,
                tiles='OpenStreetMap'
            )
            
            # Add healthcare facilities
            for facility in self.facilities:
                # Color code by facility type
                color = {
                    'Hospital': 'red',
                    'Clinic': 'blue',
                    'Urgent Care': 'orange',
                    'Specialty': 'green'
                }.get(facility.facility_type, 'gray')
                
                folium.Marker(
                    location=[facility.latitude, facility.longitude],
                    popup=folium.Popup(
                        f"<b>{facility.name}</b><br>"
                        f"Type: {facility.facility_type}<br>"
                        f"Capacity: {facility.capacity}<br>"
                        f"Rating: {facility.quality_rating}/5.0<br>"
                        f"Services: {', '.join(facility.services[:3])}",
                        max_width=300
                    ),
                    tooltip=facility.name,
                    icon=folium.Icon(color=color, icon='plus', prefix='fa')
                ).add_to(m)
            
            # Add accessibility heatmap if data available
            if self.accessibility_scores and 'data' in self.accessibility_scores:
                heat_data = [
                    [point['latitude'], point['longitude'], point['accessibility_score']]
                    for point in self.accessibility_scores['data']
                    if point['accessibility_score'] > 0
                ]
                
                if heat_data:
                    plugins.HeatMap(
                        heat_data,
                        name='Healthcare Accessibility',
                        radius=20,
                        blur=15,
                        max_zoom=1
                    ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {e}")
            return folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    def analyze_network_connectivity(self) -> Dict:
        """Analyze healthcare network connectivity and referral patterns"""
        try:
            # Create network graph
            G = nx.Graph()
            
            # Add facilities as nodes
            for i, facility in enumerate(self.facilities):
                G.add_node(i, 
                          name=facility.name,
                          type=facility.facility_type,
                          capacity=facility.capacity,
                          rating=facility.quality_rating)
            
            # Add edges based on proximity and service compatibility
            for i in range(len(self.facilities)):
                for j in range(i + 1, len(self.facilities)):
                    facility_i = self.facilities[i]
                    facility_j = self.facilities[j]
                    
                    # Calculate distance
                    distance = geodesic(
                        (facility_i.latitude, facility_i.longitude),
                        (facility_j.latitude, facility_j.longitude)
                    ).kilometers
                    
                    # Add edge if facilities are within reasonable distance
                    if distance < 200:  # 200 km threshold
                        # Weight based on distance and service overlap
                        service_overlap = len(set(facility_i.services) & set(facility_j.services))
                        weight = 1 / (1 + distance / 100) * (1 + service_overlap / 10)
                        G.add_edge(i, j, weight=weight, distance=distance)
            
            # Calculate network metrics
            connectivity_metrics = {
                'total_nodes': G.number_of_nodes(),
                'total_edges': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G) if G.number_of_nodes() > 0 else 0,
                'connected_components': nx.number_connected_components(G)
            }
            
            # Find central facilities
            if G.number_of_nodes() > 0:
                centrality = nx.degree_centrality(G)
                most_central = max(centrality.items(), key=lambda x: x[1])
                connectivity_metrics['most_central_facility'] = {
                    'name': self.facilities[most_central[0]].name,
                    'centrality_score': most_central[1]
                }
            
            return connectivity_metrics
            
        except Exception as e:
            logger.error(f"Error in network connectivity analysis: {e}")
            return {}
    
    def generate_accessibility_report(self) -> Dict:
        """Generate comprehensive accessibility analysis report"""
        try:
            # Sample population centers for analysis
            population_centers = [
                (40.7128, -74.0060),  # New York
                (34.0522, -118.2437), # Los Angeles
                (41.8781, -87.6298),  # Chicago
                (29.7604, -95.3698),  # Houston
                (33.4484, -112.0740)  # Phoenix
            ]
            
            # Load facilities and calculate accessibility
            self.load_healthcare_facilities()
            accessibility_data = self.calculate_accessibility_scores(population_centers)
            desert_analysis = self.identify_healthcare_deserts()
            network_analysis = self.analyze_network_connectivity()
            
            # Perform clustering analysis
            if accessibility_data and 'data' in accessibility_data:
                clustering_results = self.perform_spatial_clustering(accessibility_data['data'])
            else:
                clustering_results = {}
            
            report = {
                'accessibility_summary': accessibility_data.get('summary', {}),
                'healthcare_deserts': desert_analysis,
                'network_connectivity': network_analysis,
                'spatial_clustering': clustering_results,
                'recommendations': self._generate_recommendations(accessibility_data, desert_analysis),
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating accessibility report: {e}")
            return {}
    
    def _generate_recommendations(self, accessibility_data: Dict, desert_analysis: Dict) -> List[str]:
        """Generate policy and infrastructure recommendations"""
        recommendations = []
        
        try:
            if accessibility_data and 'summary' in accessibility_data:
                avg_accessibility = accessibility_data['summary'].get('avg_accessibility', 0)
                
                if avg_accessibility < 0.3:
                    recommendations.append("Consider establishing mobile health clinics in underserved areas")
                    recommendations.append("Implement telemedicine programs to improve access")
                
                if avg_accessibility < 0.5:
                    recommendations.append("Invest in transportation infrastructure to healthcare facilities")
                    recommendations.append("Develop community health worker programs")
            
            if desert_analysis:
                desert_count = desert_analysis.get('total_desert_areas', 0)
                high_severity = desert_analysis.get('high_severity_count', 0)
                
                if desert_count > 0:
                    recommendations.append(f"Address {desert_count} identified healthcare desert areas")
                
                if high_severity > 0:
                    recommendations.append(f"Prioritize {high_severity} high-severity healthcare deserts for immediate intervention")
                    recommendations.append("Consider establishing satellite clinics or urgent care centers")
            
            if not recommendations:
                recommendations.append("Healthcare accessibility appears adequate in analyzed areas")
                recommendations.append("Continue monitoring and maintain current service levels")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to data limitations")
        
        return recommendations

class HealthcareGISProcessor:
    """Advanced GIS processing for healthcare spatial analysis"""
    
    def __init__(self):
        self.spatial_analyzer = AdvancedGeospatialAnalyzer()
    
    def process_catchment_areas(self, facilities: List[HealthcareFacility], population_density: Dict) -> Dict:
        """Calculate and analyze healthcare facility catchment areas"""
        try:
            catchment_data = {}
            
            for facility in facilities:
                # Define catchment area based on facility type and capacity
                base_radius = {
                    'Hospital': 50,      # 50 km for hospitals
                    'Clinic': 25,       # 25 km for clinics
                    'Urgent Care': 15,  # 15 km for urgent care
                    'Specialty': 75     # 75 km for specialty centers
                }.get(facility.facility_type, 30)
                
                # Adjust radius based on capacity
                capacity_factor = min(facility.capacity / 500, 2.0)  # Cap at 2x
                adjusted_radius = base_radius * capacity_factor
                
                catchment_data[facility.name] = {
                    'center': (facility.latitude, facility.longitude),
                    'radius_km': adjusted_radius,
                    'estimated_population': self._estimate_catchment_population(
                        facility.latitude, facility.longitude, adjusted_radius, population_density
                    ),
                    'facility_type': facility.facility_type,
                    'capacity': facility.capacity
                }
            
            return catchment_data
            
        except Exception as e:
            logger.error(f"Error processing catchment areas: {e}")
            return {}
    
    def _estimate_catchment_population(self, lat: float, lon: float, radius_km: float, 
                                     population_density: Dict) -> int:
        """Estimate population within catchment area"""
        try:
            # Simplified population estimation
            # In a real implementation, this would use actual census data
            area_km2 = np.pi * (radius_km ** 2)
            avg_density = 100  # people per km2 (placeholder)
            estimated_population = int(area_km2 * avg_density)
            return estimated_population
            
        except Exception as e:
            logger.error(f"Error estimating catchment population: {e}")
            return 0
    
    def analyze_service_gaps(self, facilities: List[HealthcareFacility], 
                           required_services: List[str]) -> Dict:
        """Analyze gaps in healthcare service coverage"""
        try:
            service_coverage = {}
            
            # Initialize service coverage tracking
            for service in required_services:
                service_coverage[service] = {
                    'facilities_offering': [],
                    'geographic_coverage': [],
                    'capacity_total': 0
                }
            
            # Analyze each facility's service offerings
            for facility in facilities:
                for service in facility.services:
                    if service in service_coverage:
                        service_coverage[service]['facilities_offering'].append(facility.name)
                        service_coverage[service]['geographic_coverage'].append(
                            (facility.latitude, facility.longitude)
                        )
                        service_coverage[service]['capacity_total'] += facility.capacity
            
            # Identify service gaps
            gaps_analysis = {}
            for service, coverage in service_coverage.items():
                facility_count = len(coverage['facilities_offering'])
                total_capacity = coverage['capacity_total']
                
                gaps_analysis[service] = {
                    'facility_count': facility_count,
                    'total_capacity': total_capacity,
                    'coverage_level': 'High' if facility_count >= 3 else 'Medium' if facility_count >= 1 else 'Low',
                    'geographic_spread': self._calculate_geographic_spread(coverage['geographic_coverage'])
                }
            
            return {
                'service_coverage': service_coverage,
                'gaps_analysis': gaps_analysis,
                'critical_gaps': [
                    service for service, analysis in gaps_analysis.items()
                    if analysis['coverage_level'] == 'Low'
                ]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing service gaps: {e}")
            return {}
    
    def _calculate_geographic_spread(self, coordinates: List[Tuple[float, float]]) -> float:
        """Calculate geographic spread of service locations"""
        try:
            if len(coordinates) < 2:
                return 0.0
            
            # Calculate average distance between all pairs
            distances = []
            for i in range(len(coordinates)):
                for j in range(i + 1, len(coordinates)):
                    distance = geodesic(coordinates[i], coordinates[j]).kilometers
                    distances.append(distance)
            
            return np.mean(distances) if distances else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating geographic spread: {e}")
            return 0.0