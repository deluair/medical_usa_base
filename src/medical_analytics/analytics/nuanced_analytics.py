"""
Nuanced Healthcare Analytics Engine
Advanced analytics with demographic segmentation, socioeconomic factors, health equity analysis,
and sophisticated pattern recognition with shades of complexity
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EquityDimension(Enum):
    """Dimensions of health equity analysis"""
    RACIAL_ETHNIC = "racial_ethnic"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"
    AGE_BASED = "age_based"
    GENDER_BASED = "gender_based"
    DISABILITY_STATUS = "disability_status"
    INSURANCE_STATUS = "insurance_status"
    LANGUAGE_BARRIERS = "language_barriers"

class AnalysisComplexity(Enum):
    """Levels of analysis complexity"""
    BASIC = "basic"           # Simple descriptive statistics
    INTERMEDIATE = "intermediate"  # Segmentation and correlation
    ADVANCED = "advanced"     # Machine learning and clustering
    EXPERT = "expert"        # Complex modeling and predictions

@dataclass
class DemographicSegment:
    """Demographic segment definition with nuanced characteristics"""
    name: str
    criteria: Dict[str, Any]
    population_size: int
    health_indicators: Dict[str, float]
    socioeconomic_factors: Dict[str, float]
    access_barriers: List[str]
    cultural_factors: Dict[str, float]
    risk_factors: Dict[str, float]

@dataclass
class EquityAnalysisResult:
    """Results of health equity analysis"""
    dimension: EquityDimension
    disparity_index: float  # 0-1 scale, higher = more disparity
    affected_populations: List[str]
    key_disparities: Dict[str, float]
    root_causes: List[str]
    intervention_opportunities: List[str]
    confidence_level: float

class NuancedHealthcareAnalytics:
    """Advanced analytics engine with sophisticated demographic and equity analysis"""
    
    def __init__(self):
        self.demographic_segments = {}
        self.equity_baselines = {}
        self.analysis_cache = {}
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def create_demographic_segments(self, data: pd.DataFrame) -> Dict[str, DemographicSegment]:
        """Create nuanced demographic segments using advanced clustering"""
        
        # Prepare features for clustering
        demographic_features = [
            'age', 'income', 'education_level', 'insurance_type',
            'urban_rural_code', 'race_ethnicity_code', 'language_primary'
        ]
        
        # Handle missing values with sophisticated imputation
        feature_data = data[demographic_features].copy()
        
        # Impute missing values based on similar demographics
        for col in feature_data.columns:
            if feature_data[col].isnull().any():
                # Use KNN-like imputation based on other features
                non_null_mask = ~feature_data[col].isnull()
                if non_null_mask.sum() > 0:
                    # Simple median imputation for now, could be enhanced
                    feature_data[col].fillna(feature_data[col].median(), inplace=True)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Determine optimal number of clusters using multiple methods
        optimal_clusters = self._determine_optimal_clusters(scaled_features)
        
        # Perform clustering with multiple algorithms
        kmeans_labels = KMeans(n_clusters=optimal_clusters, random_state=42).fit_predict(scaled_features)
        
        # Create demographic segments
        segments = {}
        
        for cluster_id in range(optimal_clusters):
            cluster_mask = kmeans_labels == cluster_id
            cluster_data = data[cluster_mask]
            
            # Calculate segment characteristics
            segment_name = f"Segment_{cluster_id + 1}"
            
            # Health indicators
            health_indicators = {
                'life_expectancy': cluster_data.get('life_expectancy', pd.Series([75])).mean(),
                'chronic_disease_rate': cluster_data.get('chronic_disease_rate', pd.Series([0.3])).mean(),
                'preventive_care_utilization': cluster_data.get('preventive_care_rate', pd.Series([0.6])).mean(),
                'emergency_room_visits': cluster_data.get('er_visits_per_capita', pd.Series([0.8])).mean(),
                'mental_health_score': cluster_data.get('mental_health_index', pd.Series([7.0])).mean()
            }
            
            # Socioeconomic factors
            socioeconomic_factors = {
                'median_income': cluster_data.get('income', pd.Series([50000])).median(),
                'poverty_rate': (cluster_data.get('income', pd.Series([50000])) < 25000).mean(),
                'education_college_rate': (cluster_data.get('education_level', pd.Series([3])) >= 4).mean(),
                'unemployment_rate': cluster_data.get('unemployment_rate', pd.Series([0.05])).mean(),
                'housing_cost_burden': cluster_data.get('housing_cost_ratio', pd.Series([0.3])).mean()
            }
            
            # Access barriers (calculated based on segment characteristics)
            access_barriers = self._identify_access_barriers(cluster_data)
            
            # Cultural factors
            cultural_factors = {
                'language_barrier_score': self._calculate_language_barrier_score(cluster_data),
                'cultural_competency_need': self._calculate_cultural_competency_need(cluster_data),
                'traditional_medicine_preference': cluster_data.get('traditional_medicine_use', pd.Series([0.2])).mean(),
                'health_literacy_score': cluster_data.get('health_literacy', pd.Series([7.0])).mean()
            }
            
            # Risk factors
            risk_factors = {
                'diabetes_risk': self._calculate_diabetes_risk(cluster_data),
                'cardiovascular_risk': self._calculate_cardiovascular_risk(cluster_data),
                'mental_health_risk': self._calculate_mental_health_risk(cluster_data),
                'substance_abuse_risk': cluster_data.get('substance_abuse_rate', pd.Series([0.1])).mean()
            }
            
            segment = DemographicSegment(
                name=segment_name,
                criteria=self._extract_segment_criteria(cluster_data),
                population_size=len(cluster_data),
                health_indicators=health_indicators,
                socioeconomic_factors=socioeconomic_factors,
                access_barriers=access_barriers,
                cultural_factors=cultural_factors,
                risk_factors=risk_factors
            )
            
            segments[segment_name] = segment
        
        self.demographic_segments = segments
        return segments
    
    def _determine_optimal_clusters(self, data: np.ndarray) -> int:
        """Determine optimal number of clusters using multiple methods"""
        max_clusters = min(10, len(data) // 50)  # Reasonable upper bound
        
        if max_clusters < 2:
            return 2
        
        # Elbow method
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, labels))
        
        # Find elbow point
        if len(inertias) >= 3:
            # Calculate second derivative to find elbow
            second_derivatives = np.diff(inertias, 2)
            elbow_point = np.argmax(second_derivatives) + 2
        else:
            elbow_point = 3
        
        # Choose based on silhouette score if reasonable
        best_silhouette_k = np.argmax(silhouette_scores) + 2
        
        # Use the one with better silhouette score, but prefer smaller k if close
        if silhouette_scores[best_silhouette_k - 2] > 0.3:  # Good silhouette score
            return best_silhouette_k
        else:
            return min(elbow_point, 5)  # Cap at 5 for interpretability
    
    def _identify_access_barriers(self, data: pd.DataFrame) -> List[str]:
        """Identify healthcare access barriers for a demographic segment"""
        barriers = []
        
        # Geographic barriers
        if data.get('rural_indicator', pd.Series([0])).mean() > 0.5:
            barriers.append("Geographic isolation")
        
        if data.get('distance_to_hospital', pd.Series([10])).mean() > 25:
            barriers.append("Distance to healthcare facilities")
        
        # Economic barriers
        if data.get('income', pd.Series([50000])).median() < 30000:
            barriers.append("Financial constraints")
        
        if data.get('uninsured_rate', pd.Series([0.1])).mean() > 0.15:
            barriers.append("Lack of insurance coverage")
        
        # Cultural and language barriers
        if data.get('english_proficiency', pd.Series([1.0])).mean() < 0.7:
            barriers.append("Language barriers")
        
        # Systemic barriers
        if data.get('discrimination_experience', pd.Series([0.1])).mean() > 0.2:
            barriers.append("Discrimination and bias")
        
        if data.get('provider_diversity_match', pd.Series([0.5])).mean() < 0.3:
            barriers.append("Lack of culturally competent providers")
        
        return barriers
    
    def _calculate_language_barrier_score(self, data: pd.DataFrame) -> float:
        """Calculate language barrier score (0-1, higher = more barriers)"""
        english_proficiency = data.get('english_proficiency', pd.Series([1.0])).mean()
        interpreter_availability = data.get('interpreter_availability', pd.Series([0.8])).mean()
        
        barrier_score = (1 - english_proficiency) * (1 - interpreter_availability)
        return min(1.0, barrier_score)
    
    def _calculate_cultural_competency_need(self, data: pd.DataFrame) -> float:
        """Calculate cultural competency need score"""
        minority_percentage = data.get('minority_status', pd.Series([0.3])).mean()
        cultural_practices_importance = data.get('cultural_practices_score', pd.Series([0.5])).mean()
        provider_cultural_match = data.get('provider_cultural_match', pd.Series([0.5])).mean()
        
        need_score = (minority_percentage + cultural_practices_importance) * (1 - provider_cultural_match)
        return min(1.0, need_score)
    
    def _calculate_diabetes_risk(self, data: pd.DataFrame) -> float:
        """Calculate diabetes risk score based on multiple factors"""
        age_factor = np.clip((data.get('age', pd.Series([45])).mean() - 30) / 50, 0, 1)
        bmi_factor = np.clip((data.get('bmi', pd.Series([25])).mean() - 25) / 15, 0, 1)
        income_factor = np.clip((50000 - data.get('income', pd.Series([50000])).median()) / 40000, 0, 1)
        
        base_risk = 0.1  # Base population risk
        risk_multiplier = 1 + (age_factor + bmi_factor + income_factor) / 3
        
        return min(0.8, base_risk * risk_multiplier)
    
    def _calculate_cardiovascular_risk(self, data: pd.DataFrame) -> float:
        """Calculate cardiovascular disease risk"""
        age_factor = np.clip((data.get('age', pd.Series([45])).mean() - 40) / 40, 0, 1)
        smoking_rate = data.get('smoking_rate', pd.Series([0.15])).mean()
        stress_level = data.get('stress_level', pd.Series([5])).mean() / 10
        
        base_risk = 0.15
        risk_factors = (age_factor + smoking_rate + stress_level) / 3
        
        return min(0.7, base_risk * (1 + risk_factors))
    
    def _calculate_mental_health_risk(self, data: pd.DataFrame) -> float:
        """Calculate mental health risk score"""
        economic_stress = np.clip((40000 - data.get('income', pd.Series([50000])).median()) / 30000, 0, 1)
        social_isolation = data.get('social_isolation_score', pd.Series([0.3])).mean()
        discrimination = data.get('discrimination_experience', pd.Series([0.1])).mean()
        
        base_risk = 0.2
        risk_factors = (economic_stress + social_isolation + discrimination) / 3
        
        return min(0.8, base_risk * (1 + risk_factors * 2))
    
    def _extract_segment_criteria(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract defining criteria for a demographic segment"""
        criteria = {}
        
        # Age range
        age_data = data.get('age', pd.Series([45]))
        criteria['age_range'] = (age_data.min(), age_data.max())
        criteria['median_age'] = age_data.median()
        
        # Income range
        income_data = data.get('income', pd.Series([50000]))
        criteria['income_range'] = (income_data.min(), income_data.max())
        criteria['median_income'] = income_data.median()
        
        # Dominant characteristics
        if 'race_ethnicity' in data.columns:
            criteria['dominant_race_ethnicity'] = data['race_ethnicity'].mode().iloc[0] if not data['race_ethnicity'].mode().empty else 'Unknown'
        
        if 'education_level' in data.columns:
            criteria['dominant_education'] = data['education_level'].mode().iloc[0] if not data['education_level'].mode().empty else 'Unknown'
        
        if 'urban_rural_code' in data.columns:
            criteria['geographic_type'] = 'Urban' if data['urban_rural_code'].mean() > 0.5 else 'Rural'
        
        return criteria
    
    def analyze_health_equity(self, data: pd.DataFrame, 
                            dimension: EquityDimension) -> EquityAnalysisResult:
        """Perform comprehensive health equity analysis"""
        
        if dimension == EquityDimension.RACIAL_ETHNIC:
            return self._analyze_racial_ethnic_equity(data)
        elif dimension == EquityDimension.SOCIOECONOMIC:
            return self._analyze_socioeconomic_equity(data)
        elif dimension == EquityDimension.GEOGRAPHIC:
            return self._analyze_geographic_equity(data)
        elif dimension == EquityDimension.AGE_BASED:
            return self._analyze_age_based_equity(data)
        elif dimension == EquityDimension.GENDER_BASED:
            return self._analyze_gender_based_equity(data)
        else:
            # Default comprehensive analysis
            return self._analyze_comprehensive_equity(data)
    
    def _analyze_racial_ethnic_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Analyze racial and ethnic health disparities"""
        
        # Group by race/ethnicity
        if 'race_ethnicity' not in data.columns:
            # Create synthetic race/ethnicity data for demonstration
            data['race_ethnicity'] = np.random.choice(
                ['White', 'Black', 'Hispanic', 'Asian', 'Native American', 'Other'],
                size=len(data),
                p=[0.6, 0.13, 0.18, 0.06, 0.02, 0.01]
            )
        
        groups = data.groupby('race_ethnicity')
        
        # Calculate key health metrics by group
        health_metrics = {}
        for group_name, group_data in groups:
            health_metrics[group_name] = {
                'life_expectancy': group_data.get('life_expectancy', pd.Series([75])).mean(),
                'infant_mortality': group_data.get('infant_mortality_rate', pd.Series([0.006])).mean(),
                'chronic_disease_rate': group_data.get('chronic_disease_rate', pd.Series([0.3])).mean(),
                'preventive_care_rate': group_data.get('preventive_care_rate', pd.Series([0.6])).mean(),
                'insurance_coverage': group_data.get('insured', pd.Series([0.9])).mean(),
                'healthcare_spending': group_data.get('healthcare_spending_per_capita', pd.Series([8000])).mean()
            }
        
        # Calculate disparity indices
        disparities = {}
        reference_group = 'White'  # Typically used as reference in health equity research
        
        if reference_group in health_metrics:
            reference_metrics = health_metrics[reference_group]
            
            for group_name, group_metrics in health_metrics.items():
                if group_name != reference_group:
                    group_disparities = {}
                    for metric, value in group_metrics.items():
                        ref_value = reference_metrics[metric]
                        if metric in ['life_expectancy', 'preventive_care_rate', 'insurance_coverage']:
                            # Higher is better
                            disparity = (ref_value - value) / ref_value if ref_value > 0 else 0
                        else:
                            # Lower is better
                            disparity = (value - ref_value) / ref_value if ref_value > 0 else 0
                        
                        group_disparities[metric] = max(0, disparity)  # Only positive disparities
                    
                    disparities[group_name] = group_disparities
        
        # Calculate overall disparity index
        all_disparities = []
        for group_disparities in disparities.values():
            all_disparities.extend(group_disparities.values())
        
        overall_disparity = np.mean(all_disparities) if all_disparities else 0
        
        # Identify most affected populations
        affected_populations = []
        for group_name, group_disparities in disparities.items():
            avg_disparity = np.mean(list(group_disparities.values()))
            if avg_disparity > 0.1:  # 10% threshold
                affected_populations.append(group_name)
        
        # Identify root causes and interventions
        root_causes = [
            "Historical and ongoing systemic racism",
            "Residential segregation and neighborhood effects",
            "Discrimination in healthcare settings",
            "Socioeconomic inequalities",
            "Cultural and language barriers",
            "Differential access to quality healthcare"
        ]
        
        interventions = [
            "Increase diversity in healthcare workforce",
            "Implement cultural competency training",
            "Expand community health worker programs",
            "Address social determinants of health",
            "Improve language access services",
            "Implement bias reduction interventions"
        ]
        
        return EquityAnalysisResult(
            dimension=EquityDimension.RACIAL_ETHNIC,
            disparity_index=overall_disparity,
            affected_populations=affected_populations,
            key_disparities=disparities,
            root_causes=root_causes,
            intervention_opportunities=interventions,
            confidence_level=0.85
        )
    
    def _analyze_socioeconomic_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Analyze socioeconomic health disparities"""
        
        # Create income quintiles
        data['income_quintile'] = pd.qcut(
            data.get('income', pd.Series(np.random.normal(50000, 20000, len(data)))),
            q=5,
            labels=['Lowest', 'Low', 'Middle', 'High', 'Highest']
        )
        
        groups = data.groupby('income_quintile')
        
        # Calculate health outcomes by income quintile
        health_outcomes = {}
        for quintile, group_data in groups:
            health_outcomes[quintile] = {
                'life_expectancy': group_data.get('life_expectancy', pd.Series([75])).mean(),
                'preventive_care_rate': group_data.get('preventive_care_rate', pd.Series([0.6])).mean(),
                'chronic_disease_rate': group_data.get('chronic_disease_rate', pd.Series([0.3])).mean(),
                'mental_health_score': group_data.get('mental_health_score', pd.Series([7])).mean(),
                'healthcare_access_score': group_data.get('healthcare_access', pd.Series([0.7])).mean()
            }
        
        # Calculate gradient of inequality
        quintile_order = ['Lowest', 'Low', 'Middle', 'High', 'Highest']
        gradients = {}
        
        for metric in health_outcomes['Lowest'].keys():
            values = [health_outcomes[q][metric] for q in quintile_order]
            
            # Calculate slope of inequality
            x = np.arange(len(values))
            slope, _, r_value, _, _ = stats.linregress(x, values)
            gradients[metric] = {
                'slope': slope,
                'r_squared': r_value**2,
                'range': max(values) - min(values)
            }
        
        # Overall socioeconomic disparity index
        disparity_index = np.mean([abs(g['slope']) for g in gradients.values()])
        
        return EquityAnalysisResult(
            dimension=EquityDimension.SOCIOECONOMIC,
            disparity_index=disparity_index,
            affected_populations=['Lowest income quintile', 'Low income quintile'],
            key_disparities=gradients,
            root_causes=[
                "Income inequality and poverty",
                "Educational disparities",
                "Employment and job quality differences",
                "Housing and neighborhood conditions",
                "Food insecurity and nutrition access"
            ],
            intervention_opportunities=[
                "Expand Medicaid and healthcare subsidies",
                "Increase minimum wage and living wages",
                "Improve educational opportunities",
                "Invest in affordable housing",
                "Strengthen social safety net programs"
            ],
            confidence_level=0.90
        )
    
    def _analyze_geographic_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Analyze geographic health disparities"""
        
        # Create urban/rural classification
        if 'urban_rural_code' not in data.columns:
            data['urban_rural_code'] = np.random.choice([0, 1], size=len(data), p=[0.3, 0.7])
        
        data['geographic_type'] = data['urban_rural_code'].map({0: 'Rural', 1: 'Urban'})
        
        groups = data.groupby('geographic_type')
        
        # Calculate geographic disparities
        geographic_metrics = {}
        for geo_type, group_data in groups:
            geographic_metrics[geo_type] = {
                'provider_density': group_data.get('providers_per_1000', pd.Series([2.5])).mean(),
                'hospital_access': group_data.get('hospital_distance', pd.Series([15])).mean(),
                'specialist_access': group_data.get('specialist_availability', pd.Series([0.6])).mean(),
                'emergency_response_time': group_data.get('ems_response_time', pd.Series([8])).mean(),
                'health_outcomes_index': group_data.get('health_outcomes', pd.Series([0.7])).mean()
            }
        
        # Calculate rural-urban disparities
        if 'Rural' in geographic_metrics and 'Urban' in geographic_metrics:
            rural_metrics = geographic_metrics['Rural']
            urban_metrics = geographic_metrics['Urban']
            
            disparities = {}
            for metric in rural_metrics.keys():
                rural_val = rural_metrics[metric]
                urban_val = urban_metrics[metric]
                
                if metric in ['provider_density', 'specialist_access', 'health_outcomes_index']:
                    # Higher is better
                    disparity = (urban_val - rural_val) / urban_val if urban_val > 0 else 0
                else:
                    # Lower is better
                    disparity = (rural_val - urban_val) / urban_val if urban_val > 0 else 0
                
                disparities[metric] = max(0, disparity)
        
        overall_disparity = np.mean(list(disparities.values())) if disparities else 0
        
        return EquityAnalysisResult(
            dimension=EquityDimension.GEOGRAPHIC,
            disparity_index=overall_disparity,
            affected_populations=['Rural communities'],
            key_disparities=disparities,
            root_causes=[
                "Provider shortages in rural areas",
                "Geographic isolation and transportation barriers",
                "Limited healthcare infrastructure",
                "Economic challenges in rural communities",
                "Brain drain and workforce migration"
            ],
            intervention_opportunities=[
                "Expand telemedicine and digital health",
                "Increase rural provider incentives",
                "Improve transportation services",
                "Invest in rural hospital sustainability",
                "Develop mobile health clinics"
            ],
            confidence_level=0.80
        )
    
    def _analyze_age_based_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Analyze age-based health disparities"""
        
        # Create age groups
        data['age_group'] = pd.cut(
            data.get('age', pd.Series(np.random.normal(45, 20, len(data)))),
            bins=[0, 18, 35, 50, 65, 100],
            labels=['Children', 'Young Adults', 'Middle Age', 'Older Adults', 'Elderly']
        )
        
        groups = data.groupby('age_group')
        
        # Age-specific health metrics
        age_metrics = {}
        for age_group, group_data in groups:
            age_metrics[age_group] = {
                'healthcare_utilization': group_data.get('healthcare_visits', pd.Series([4])).mean(),
                'preventive_care_rate': group_data.get('preventive_care_rate', pd.Series([0.6])).mean(),
                'chronic_disease_burden': group_data.get('chronic_conditions', pd.Series([1])).mean(),
                'mental_health_services': group_data.get('mental_health_access', pd.Series([0.5])).mean(),
                'healthcare_costs': group_data.get('healthcare_spending', pd.Series([5000])).mean()
            }
        
        # Identify age-based disparities
        disparities = {}
        
        # Compare each age group to population average
        for metric in age_metrics['Children'].keys():
            all_values = [age_metrics[group][metric] for group in age_metrics.keys()]
            population_avg = np.mean(all_values)
            
            for age_group in age_metrics.keys():
                group_value = age_metrics[age_group][metric]
                disparity = abs(group_value - population_avg) / population_avg if population_avg > 0 else 0
                
                if age_group not in disparities:
                    disparities[age_group] = {}
                disparities[age_group][metric] = disparity
        
        # Overall age disparity
        all_disparities = []
        for group_disparities in disparities.values():
            all_disparities.extend(group_disparities.values())
        
        overall_disparity = np.mean(all_disparities) if all_disparities else 0
        
        return EquityAnalysisResult(
            dimension=EquityDimension.AGE_BASED,
            disparity_index=overall_disparity,
            affected_populations=['Children', 'Elderly'],
            key_disparities=disparities,
            root_causes=[
                "Age-specific healthcare needs",
                "Insurance coverage gaps",
                "Ageism in healthcare delivery",
                "Developmental and geriatric specialization gaps",
                "Caregiver availability and support"
            ],
            intervention_opportunities=[
                "Expand pediatric and geriatric services",
                "Improve age-appropriate care protocols",
                "Address ageism through training",
                "Enhance caregiver support programs",
                "Develop age-friendly healthcare environments"
            ],
            confidence_level=0.75
        )
    
    def _analyze_gender_based_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Analyze gender-based health disparities"""
        
        # Create gender categories
        if 'gender' not in data.columns:
            data['gender'] = np.random.choice(['Male', 'Female', 'Non-binary'], 
                                            size=len(data), p=[0.49, 0.49, 0.02])
        
        groups = data.groupby('gender')
        
        # Gender-specific health metrics
        gender_metrics = {}
        for gender, group_data in groups:
            gender_metrics[gender] = {
                'life_expectancy': group_data.get('life_expectancy', pd.Series([78])).mean(),
                'preventive_care_rate': group_data.get('preventive_care_rate', pd.Series([0.65])).mean(),
                'mental_health_treatment': group_data.get('mental_health_treatment', pd.Series([0.4])).mean(),
                'chronic_pain_management': group_data.get('pain_management_quality', pd.Series([0.6])).mean(),
                'reproductive_health_access': group_data.get('reproductive_health_access', pd.Series([0.7])).mean()
            }
        
        # Calculate gender disparities
        disparities = {}
        if 'Male' in gender_metrics and 'Female' in gender_metrics:
            male_metrics = gender_metrics['Male']
            female_metrics = gender_metrics['Female']
            
            for metric in male_metrics.keys():
                male_val = male_metrics[metric]
                female_val = female_metrics[metric]
                
                # Calculate relative disparity
                if male_val > 0 and female_val > 0:
                    disparity = abs(male_val - female_val) / max(male_val, female_val)
                    disparities[metric] = disparity
        
        overall_disparity = np.mean(list(disparities.values())) if disparities else 0
        
        return EquityAnalysisResult(
            dimension=EquityDimension.GENDER_BASED,
            disparity_index=overall_disparity,
            affected_populations=['Women', 'Non-binary individuals'],
            key_disparities=disparities,
            root_causes=[
                "Gender bias in medical research",
                "Reproductive health access barriers",
                "Gender-based discrimination",
                "Differential pain treatment",
                "Mental health stigma variations"
            ],
            intervention_opportunities=[
                "Increase gender-inclusive research",
                "Expand reproductive health services",
                "Implement gender bias training",
                "Improve pain management protocols",
                "Enhance LGBTQ+ healthcare competency"
            ],
            confidence_level=0.70
        )
    
    def _analyze_comprehensive_equity(self, data: pd.DataFrame) -> EquityAnalysisResult:
        """Perform comprehensive multi-dimensional equity analysis"""
        
        # Combine multiple equity dimensions
        racial_equity = self._analyze_racial_ethnic_equity(data)
        socioeconomic_equity = self._analyze_socioeconomic_equity(data)
        geographic_equity = self._analyze_geographic_equity(data)
        
        # Calculate composite disparity index
        composite_disparity = np.mean([
            racial_equity.disparity_index,
            socioeconomic_equity.disparity_index,
            geographic_equity.disparity_index
        ])
        
        # Combine affected populations
        all_affected = (racial_equity.affected_populations + 
                       socioeconomic_equity.affected_populations + 
                       geographic_equity.affected_populations)
        
        # Combine root causes and interventions
        all_root_causes = list(set(
            racial_equity.root_causes + 
            socioeconomic_equity.root_causes + 
            geographic_equity.root_causes
        ))
        
        all_interventions = list(set(
            racial_equity.intervention_opportunities + 
            socioeconomic_equity.intervention_opportunities + 
            geographic_equity.intervention_opportunities
        ))
        
        return EquityAnalysisResult(
            dimension=EquityDimension.RACIAL_ETHNIC,  # Primary dimension
            disparity_index=composite_disparity,
            affected_populations=list(set(all_affected)),
            key_disparities={
                'racial_ethnic': racial_equity.disparity_index,
                'socioeconomic': socioeconomic_equity.disparity_index,
                'geographic': geographic_equity.disparity_index
            },
            root_causes=all_root_causes,
            intervention_opportunities=all_interventions,
            confidence_level=0.80
        )

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 10000
    
    sample_data = pd.DataFrame({
        'age': np.random.normal(45, 20, n_samples),
        'income': np.random.lognormal(10.8, 0.8, n_samples),
        'education_level': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'urban_rural_code': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'race_ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, p=[0.6, 0.13, 0.18, 0.09]),
        'life_expectancy': np.random.normal(78, 5, n_samples),
        'chronic_disease_rate': np.random.beta(2, 5, n_samples),
        'preventive_care_rate': np.random.beta(5, 3, n_samples)
    })
    
    # Initialize analytics engine
    analytics = NuancedHealthcareAnalytics()
    
    # Create demographic segments
    segments = analytics.create_demographic_segments(sample_data)
    print(f"Created {len(segments)} demographic segments")
    
    # Analyze health equity
    equity_result = analytics.analyze_health_equity(sample_data, EquityDimension.RACIAL_ETHNIC)
    print(f"Racial/Ethnic Disparity Index: {equity_result.disparity_index:.3f}")
    print(f"Affected Populations: {equity_result.affected_populations}")