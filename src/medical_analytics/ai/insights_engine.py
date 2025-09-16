"""
AI-Powered Healthcare Insights Engine
Advanced AI system for automated trend detection, natural language explanations,
intelligent pattern recognition, and predictive insights with nuanced understanding
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
import re
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class InsightType(Enum):
    """Types of AI-generated insights"""
    TREND_DETECTION = "trend_detection"
    ANOMALY_IDENTIFICATION = "anomaly_identification"
    PATTERN_RECOGNITION = "pattern_recognition"
    CORRELATION_DISCOVERY = "correlation_discovery"
    PREDICTIVE_FORECAST = "predictive_forecast"
    CAUSAL_INFERENCE = "causal_inference"
    RISK_ASSESSMENT = "risk_assessment"
    OPPORTUNITY_IDENTIFICATION = "opportunity_identification"

class ConfidenceLevel(Enum):
    """Confidence levels for insights"""
    VERY_HIGH = "very_high"  # 90%+
    HIGH = "high"           # 75-90%
    MEDIUM = "medium"       # 50-75%
    LOW = "low"            # 25-50%
    VERY_LOW = "very_low"  # <25%

@dataclass
class Insight:
    """AI-generated insight with natural language explanation"""
    type: InsightType
    title: str
    description: str
    natural_language_explanation: str
    confidence: ConfidenceLevel
    confidence_score: float
    supporting_evidence: List[str]
    data_points: Dict[str, Any]
    recommendations: List[str]
    impact_assessment: str
    urgency_level: int  # 1-5 scale
    affected_populations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class TrendAnalysis:
    """Comprehensive trend analysis results"""
    trend_direction: str  # "increasing", "decreasing", "stable", "cyclical"
    trend_strength: float  # 0-1 scale
    trend_significance: float  # p-value
    seasonal_component: bool
    change_points: List[int]  # Indices where trend changes
    forecast_values: List[float]
    forecast_confidence_intervals: List[Tuple[float, float]]

class AIHealthcareInsightsEngine:
    """Advanced AI engine for healthcare insights and pattern recognition"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.trend_models = {}
        self.pattern_cache = {}
        self.insight_history = []
        
        # Natural language templates
        self.language_templates = self._initialize_language_templates()
        
    def _initialize_language_templates(self) -> Dict[InsightType, Dict[str, str]]:
        """Initialize natural language templates for different insight types"""
        return {
            InsightType.TREND_DETECTION: {
                'increasing': "There is a {strength} upward trend in {metric} over the analyzed period. "
                            "The trend shows {significance} statistical significance with a {change_rate}% "
                            "change rate. This suggests {implication}.",
                'decreasing': "A {strength} downward trend has been detected in {metric}. "
                            "The decline is {significance} statistically significant with a {change_rate}% "
                            "decrease rate. This indicates {implication}.",
                'stable': "{metric} has remained relatively stable over the analyzed period, "
                         "showing minimal variation around the mean value. This stability suggests {implication}.",
                'cyclical': "A cyclical pattern has been identified in {metric} with {cycle_length} "
                           "recurring cycles. This pattern suggests {implication}."
            },
            InsightType.ANOMALY_IDENTIFICATION: {
                'outlier': "An anomalous pattern has been detected in {metric} at {time_period}. "
                          "The observed value of {value} deviates significantly from the expected range "
                          "of {expected_range}. This anomaly may indicate {potential_cause}.",
                'cluster': "A cluster of {count} anomalous data points has been identified in {metric}. "
                          "These outliers suggest {pattern_description} and may require {action_needed}."
            },
            InsightType.CORRELATION_DISCOVERY: {
                'strong_positive': "A strong positive correlation (r={correlation:.3f}) has been discovered "
                                  "between {variable1} and {variable2}. This relationship suggests that "
                                  "as {variable1} increases, {variable2} tends to increase as well. "
                                  "This finding implies {implication}.",
                'strong_negative': "A strong negative correlation (r={correlation:.3f}) exists between "
                                  "{variable1} and {variable2}. This inverse relationship indicates that "
                                  "higher values of {variable1} are associated with lower values of {variable2}. "
                                  "This suggests {implication}.",
                'moderate': "A moderate correlation (r={correlation:.3f}) has been identified between "
                           "{variable1} and {variable2}. While not as strong as other relationships, "
                           "this connection suggests {implication}."
            },
            InsightType.PREDICTIVE_FORECAST: {
                'forecast': "Based on historical patterns, {metric} is predicted to {direction} "
                           "to approximately {predicted_value} over the next {time_horizon}. "
                           "The forecast has a confidence interval of [{lower_bound}, {upper_bound}] "
                           "and suggests {implication}."
            },
            InsightType.RISK_ASSESSMENT: {
                'high_risk': "A high-risk situation has been identified in {area}. "
                            "The risk score of {risk_score} indicates {risk_description}. "
                            "Immediate attention and {recommended_action} are recommended.",
                'moderate_risk': "Moderate risk levels detected in {area} with a risk score of {risk_score}. "
                               "While not immediately critical, {preventive_measures} should be considered.",
                'low_risk': "Risk assessment indicates low risk levels in {area} (score: {risk_score}). "
                           "Current conditions are favorable, but continued monitoring is advised."
            }
        }
    
    def analyze_comprehensive_insights(self, data: pd.DataFrame, 
                                     target_columns: List[str] = None,
                                     time_column: str = None) -> List[Insight]:
        """Generate comprehensive AI insights from healthcare data"""
        
        insights = []
        
        if target_columns is None:
            # Auto-detect numeric columns for analysis
            target_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Trend Detection
        if time_column and time_column in data.columns:
            trend_insights = self._detect_trends(data, target_columns, time_column)
            insights.extend(trend_insights)
        
        # 2. Anomaly Detection
        anomaly_insights = self._detect_anomalies(data, target_columns)
        insights.extend(anomaly_insights)
        
        # 3. Pattern Recognition
        pattern_insights = self._recognize_patterns(data, target_columns)
        insights.extend(pattern_insights)
        
        # 4. Correlation Discovery
        correlation_insights = self._discover_correlations(data, target_columns)
        insights.extend(correlation_insights)
        
        # 5. Risk Assessment
        risk_insights = self._assess_risks(data, target_columns)
        insights.extend(risk_insights)
        
        # 6. Predictive Forecasting
        if time_column and time_column in data.columns:
            forecast_insights = self._generate_forecasts(data, target_columns, time_column)
            insights.extend(forecast_insights)
        
        # Sort insights by urgency and confidence
        insights.sort(key=lambda x: (x.urgency_level, x.confidence_score), reverse=True)
        
        # Store in history
        self.insight_history.extend(insights)
        
        return insights
    
    def _detect_trends(self, data: pd.DataFrame, 
                      target_columns: List[str], 
                      time_column: str) -> List[Insight]:
        """Detect trends in time series data with AI analysis"""
        
        insights = []
        
        # Sort by time
        data_sorted = data.sort_values(time_column)
        
        for column in target_columns:
            if column in data_sorted.columns and data_sorted[column].notna().sum() > 10:
                
                # Perform trend analysis
                trend_analysis = self._analyze_trend(data_sorted[column].values)
                
                # Generate natural language explanation
                strength_desc = self._get_strength_description(trend_analysis.trend_strength)
                significance_desc = self._get_significance_description(trend_analysis.trend_significance)
                
                # Calculate change rate
                if len(data_sorted[column]) > 1:
                    start_val = data_sorted[column].iloc[0]
                    end_val = data_sorted[column].iloc[-1]
                    change_rate = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                else:
                    change_rate = 0
                
                # Generate implication based on column name and trend
                implication = self._generate_trend_implication(column, trend_analysis.trend_direction, change_rate)
                
                # Create natural language explanation
                template = self.language_templates[InsightType.TREND_DETECTION][trend_analysis.trend_direction]
                explanation = template.format(
                    metric=column.replace('_', ' ').title(),
                    strength=strength_desc,
                    significance=significance_desc,
                    change_rate=abs(change_rate),
                    implication=implication
                )
                
                # Determine confidence level
                confidence_score = min(trend_analysis.trend_strength * (1 - trend_analysis.trend_significance), 1.0)
                confidence_level = self._score_to_confidence_level(confidence_score)
                
                # Create insight
                insight = Insight(
                    type=InsightType.TREND_DETECTION,
                    title=f"Trend Analysis: {column.replace('_', ' ').title()}",
                    description=f"{trend_analysis.trend_direction.title()} trend detected with {strength_desc} strength",
                    natural_language_explanation=explanation,
                    confidence=confidence_level,
                    confidence_score=confidence_score,
                    supporting_evidence=[
                        f"Trend strength: {trend_analysis.trend_strength:.3f}",
                        f"Statistical significance: p={trend_analysis.trend_significance:.4f}",
                        f"Change rate: {change_rate:.1f}%"
                    ],
                    data_points={
                        'trend_direction': trend_analysis.trend_direction,
                        'trend_strength': trend_analysis.trend_strength,
                        'change_rate': change_rate,
                        'change_points': trend_analysis.change_points
                    },
                    recommendations=self._generate_trend_recommendations(column, trend_analysis),
                    impact_assessment=self._assess_trend_impact(column, trend_analysis),
                    urgency_level=self._calculate_trend_urgency(trend_analysis),
                    affected_populations=self._identify_affected_populations(column, data)
                )
                
                insights.append(insight)
        
        return insights
    
    def _analyze_trend(self, values: np.ndarray) -> TrendAnalysis:
        """Perform comprehensive trend analysis on time series data"""
        
        # Remove NaN values
        clean_values = values[~np.isnan(values)]
        
        if len(clean_values) < 3:
            return TrendAnalysis(
                trend_direction="stable",
                trend_strength=0.0,
                trend_significance=1.0,
                seasonal_component=False,
                change_points=[],
                forecast_values=[],
                forecast_confidence_intervals=[]
            )
        
        # Linear trend analysis
        x = np.arange(len(clean_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, clean_values)
        
        # Determine trend direction
        if abs(slope) < std_err:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Calculate trend strength (based on R-squared)
        trend_strength = abs(r_value)
        
        # Detect change points using statistical methods
        change_points = self._detect_change_points(clean_values)
        
        # Check for seasonality (simple approach)
        seasonal_component = self._detect_seasonality(clean_values)
        
        # Generate forecast
        forecast_values, confidence_intervals = self._generate_simple_forecast(clean_values, periods=5)
        
        return TrendAnalysis(
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            trend_significance=p_value,
            seasonal_component=seasonal_component,
            change_points=change_points,
            forecast_values=forecast_values,
            forecast_confidence_intervals=confidence_intervals
        )
    
    def _detect_change_points(self, values: np.ndarray) -> List[int]:
        """Detect change points in time series using statistical methods"""
        
        if len(values) < 10:
            return []
        
        change_points = []
        window_size = max(5, len(values) // 10)
        
        for i in range(window_size, len(values) - window_size):
            # Compare means before and after potential change point
            before = values[i-window_size:i]
            after = values[i:i+window_size]
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.05:  # Significant difference
                change_points.append(i)
        
        # Remove nearby change points (keep only significant ones)
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] > window_size:
                filtered_change_points.append(cp)
        
        return filtered_change_points
    
    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Simple seasonality detection using autocorrelation"""
        
        if len(values) < 24:  # Need sufficient data
            return False
        
        # Calculate autocorrelation for different lags
        max_lag = min(len(values) // 4, 12)
        autocorrelations = []
        
        for lag in range(1, max_lag + 1):
            if len(values) > lag:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                if not np.isnan(corr):
                    autocorrelations.append(abs(corr))
        
        # Check if any autocorrelation is significantly high
        return max(autocorrelations) > 0.3 if autocorrelations else False
    
    def _generate_simple_forecast(self, values: np.ndarray, periods: int = 5) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate simple forecast using linear regression"""
        
        if len(values) < 3:
            return [], []
        
        # Fit linear model
        x = np.arange(len(values)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, values)
        
        # Generate forecast
        future_x = np.arange(len(values), len(values) + periods).reshape(-1, 1)
        forecast = model.predict(future_x)
        
        # Calculate prediction intervals (simple approach)
        residuals = values - model.predict(x)
        std_residual = np.std(residuals)
        
        confidence_intervals = []
        for i, pred in enumerate(forecast):
            # Increase uncertainty with distance
            uncertainty = std_residual * (1 + i * 0.1)
            lower = pred - 1.96 * uncertainty
            upper = pred + 1.96 * uncertainty
            confidence_intervals.append((lower, upper))
        
        return forecast.tolist(), confidence_intervals
    
    def _detect_anomalies(self, data: pd.DataFrame, target_columns: List[str]) -> List[Insight]:
        """Detect anomalies using AI-based methods"""
        
        insights = []
        
        # Prepare data for anomaly detection
        numeric_data = data[target_columns].select_dtypes(include=[np.number])
        
        if len(numeric_data) == 0 or len(numeric_data.columns) == 0:
            return insights
        
        # Fill NaN values
        numeric_data_filled = numeric_data.fillna(numeric_data.median())
        
        if len(numeric_data_filled) < 10:
            return insights
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data_filled)
        
        # Detect anomalies
        anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
        anomaly_scores = self.anomaly_detector.decision_function(scaled_data)
        
        # Identify anomalous points
        anomaly_indices = np.where(anomaly_labels == -1)[0]
        
        if len(anomaly_indices) > 0:
            # Analyze anomalies
            for idx in anomaly_indices[:5]:  # Limit to top 5 anomalies
                
                anomaly_row = numeric_data.iloc[idx]
                anomaly_score = anomaly_scores[idx]
                
                # Find which columns contribute most to the anomaly
                row_scaled = scaled_data[idx]
                column_contributions = np.abs(row_scaled)
                top_contributor_idx = np.argmax(column_contributions)
                top_contributor = numeric_data.columns[top_contributor_idx]
                
                # Generate explanation
                anomaly_value = anomaly_row[top_contributor]
                expected_range = self._calculate_expected_range(numeric_data[top_contributor])
                
                explanation = self.language_templates[InsightType.ANOMALY_IDENTIFICATION]['outlier'].format(
                    metric=top_contributor.replace('_', ' ').title(),
                    time_period=f"row {idx}",
                    value=f"{anomaly_value:.2f}",
                    expected_range=f"{expected_range[0]:.2f} - {expected_range[1]:.2f}",
                    potential_cause=self._infer_anomaly_cause(top_contributor, anomaly_value, expected_range)
                )
                
                # Calculate confidence
                confidence_score = min(abs(anomaly_score) / 2, 1.0)  # Normalize anomaly score
                confidence_level = self._score_to_confidence_level(confidence_score)
                
                insight = Insight(
                    type=InsightType.ANOMALY_IDENTIFICATION,
                    title=f"Anomaly Detected: {top_contributor.replace('_', ' ').title()}",
                    description=f"Unusual value detected in {top_contributor.replace('_', ' ').title()}",
                    natural_language_explanation=explanation,
                    confidence=confidence_level,
                    confidence_score=confidence_score,
                    supporting_evidence=[
                        f"Anomaly score: {anomaly_score:.3f}",
                        f"Value: {anomaly_value:.2f}",
                        f"Expected range: {expected_range[0]:.2f} - {expected_range[1]:.2f}"
                    ],
                    data_points={
                        'anomaly_index': int(idx),
                        'anomaly_score': float(anomaly_score),
                        'anomaly_value': float(anomaly_value),
                        'expected_range': expected_range
                    },
                    recommendations=self._generate_anomaly_recommendations(top_contributor, anomaly_value),
                    impact_assessment=self._assess_anomaly_impact(top_contributor, anomaly_value),
                    urgency_level=self._calculate_anomaly_urgency(anomaly_score),
                    affected_populations=self._identify_affected_populations(top_contributor, data)
                )
                
                insights.append(insight)
        
        return insights
    
    def _discover_correlations(self, data: pd.DataFrame, target_columns: List[str]) -> List[Insight]:
        """Discover significant correlations using AI analysis"""
        
        insights = []
        
        # Calculate correlation matrix
        numeric_data = data[target_columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2:
            return insights
        
        correlation_matrix = numeric_data.corr()
        
        # Find significant correlations
        significant_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if not np.isnan(corr_value) and abs(corr_value) > 0.5:  # Significant correlation threshold
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    
                    significant_correlations.append({
                        'var1': var1,
                        'var2': var2,
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                    })
        
        # Generate insights for top correlations
        significant_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        for corr_info in significant_correlations[:5]:  # Top 5 correlations
            
            # Determine correlation type
            if corr_info['correlation'] > 0:
                corr_type = 'strong_positive' if corr_info['strength'] == 'strong' else 'moderate'
            else:
                corr_type = 'strong_negative' if corr_info['strength'] == 'strong' else 'moderate'
            
            # Generate implication
            implication = self._generate_correlation_implication(
                corr_info['var1'], corr_info['var2'], corr_info['correlation']
            )
            
            # Create explanation
            template = self.language_templates[InsightType.CORRELATION_DISCOVERY][corr_type]
            explanation = template.format(
                variable1=corr_info['var1'].replace('_', ' ').title(),
                variable2=corr_info['var2'].replace('_', ' ').title(),
                correlation=corr_info['correlation'],
                implication=implication
            )
            
            # Calculate confidence
            confidence_score = abs(corr_info['correlation'])
            confidence_level = self._score_to_confidence_level(confidence_score)
            
            insight = Insight(
                type=InsightType.CORRELATION_DISCOVERY,
                title=f"Correlation: {corr_info['var1'].replace('_', ' ').title()} & {corr_info['var2'].replace('_', ' ').title()}",
                description=f"{corr_info['strength'].title()} {corr_type.replace('_', ' ')} correlation detected",
                natural_language_explanation=explanation,
                confidence=confidence_level,
                confidence_score=confidence_score,
                supporting_evidence=[
                    f"Correlation coefficient: {corr_info['correlation']:.3f}",
                    f"Correlation strength: {corr_info['strength']}",
                    f"Sample size: {len(numeric_data)}"
                ],
                data_points={
                    'variable1': corr_info['var1'],
                    'variable2': corr_info['var2'],
                    'correlation_value': corr_info['correlation'],
                    'correlation_strength': corr_info['strength']
                },
                recommendations=self._generate_correlation_recommendations(corr_info),
                impact_assessment=self._assess_correlation_impact(corr_info),
                urgency_level=self._calculate_correlation_urgency(corr_info),
                affected_populations=self._identify_affected_populations_correlation(corr_info, data)
            )
            
            insights.append(insight)
        
        return insights
    
    def _assess_risks(self, data: pd.DataFrame, target_columns: List[str]) -> List[Insight]:
        """Assess risks using AI-based risk scoring"""
        
        insights = []
        
        # Define risk indicators for healthcare
        risk_indicators = {
            'mortality_rate': {'threshold': 0.05, 'direction': 'higher'},
            'readmission_rate': {'threshold': 0.15, 'direction': 'higher'},
            'infection_rate': {'threshold': 0.10, 'direction': 'higher'},
            'wait_time': {'threshold': 60, 'direction': 'higher'},
            'satisfaction_score': {'threshold': 6.0, 'direction': 'lower'},
            'staffing_ratio': {'threshold': 0.8, 'direction': 'lower'},
            'budget_variance': {'threshold': 0.10, 'direction': 'higher'}
        }
        
        for column in target_columns:
            # Check if column matches any risk indicator
            for risk_name, risk_config in risk_indicators.items():
                if risk_name in column.lower() and column in data.columns:
                    
                    values = data[column].dropna()
                    if len(values) == 0:
                        continue
                    
                    # Calculate risk score
                    threshold = risk_config['threshold']
                    direction = risk_config['direction']
                    
                    if direction == 'higher':
                        risk_values = values[values > threshold]
                        risk_score = len(risk_values) / len(values)
                        avg_risk_value = values.mean()
                    else:
                        risk_values = values[values < threshold]
                        risk_score = len(risk_values) / len(values)
                        avg_risk_value = values.mean()
                    
                    # Determine risk level
                    if risk_score > 0.7:
                        risk_level = 'high_risk'
                        urgency = 5
                    elif risk_score > 0.3:
                        risk_level = 'moderate_risk'
                        urgency = 3
                    else:
                        risk_level = 'low_risk'
                        urgency = 1
                    
                    # Generate risk description and recommendations
                    risk_description = self._generate_risk_description(column, risk_score, avg_risk_value)
                    recommended_action = self._generate_risk_recommendations(column, risk_level)
                    
                    # Create explanation
                    template = self.language_templates[InsightType.RISK_ASSESSMENT][risk_level]
                    
                    if risk_level == 'high_risk':
                        explanation = template.format(
                            area=column.replace('_', ' ').title(),
                            risk_score=f"{risk_score:.2f}",
                            risk_description=risk_description,
                            recommended_action=recommended_action
                        )
                    elif risk_level == 'moderate_risk':
                        explanation = template.format(
                            area=column.replace('_', ' ').title(),
                            risk_score=f"{risk_score:.2f}",
                            preventive_measures=recommended_action
                        )
                    else:
                        explanation = template.format(
                            area=column.replace('_', ' ').title(),
                            risk_score=f"{risk_score:.2f}"
                        )
                    
                    # Calculate confidence
                    confidence_score = min(risk_score + 0.3, 1.0)  # Higher risk = higher confidence
                    confidence_level = self._score_to_confidence_level(confidence_score)
                    
                    insight = Insight(
                        type=InsightType.RISK_ASSESSMENT,
                        title=f"Risk Assessment: {column.replace('_', ' ').title()}",
                        description=f"{risk_level.replace('_', ' ').title()} detected",
                        natural_language_explanation=explanation,
                        confidence=confidence_level,
                        confidence_score=confidence_score,
                        supporting_evidence=[
                            f"Risk score: {risk_score:.3f}",
                            f"Threshold: {threshold}",
                            f"Average value: {avg_risk_value:.2f}",
                            f"At-risk percentage: {risk_score*100:.1f}%"
                        ],
                        data_points={
                            'risk_score': risk_score,
                            'risk_level': risk_level,
                            'threshold': threshold,
                            'average_value': avg_risk_value
                        },
                        recommendations=recommended_action.split(', '),
                        impact_assessment=self._assess_risk_impact(column, risk_score),
                        urgency_level=urgency,
                        affected_populations=self._identify_affected_populations(column, data)
                    )
                    
                    insights.append(insight)
        
        return insights
    
    # Helper methods for generating natural language content
    
    def _get_strength_description(self, strength: float) -> str:
        """Convert numeric strength to descriptive text"""
        if strength > 0.8:
            return "very strong"
        elif strength > 0.6:
            return "strong"
        elif strength > 0.4:
            return "moderate"
        elif strength > 0.2:
            return "weak"
        else:
            return "very weak"
    
    def _get_significance_description(self, p_value: float) -> str:
        """Convert p-value to significance description"""
        if p_value < 0.001:
            return "highly"
        elif p_value < 0.01:
            return "very"
        elif p_value < 0.05:
            return "statistically"
        else:
            return "not statistically"
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level enum"""
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.75:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_trend_implication(self, column: str, direction: str, change_rate: float) -> str:
        """Generate contextual implications for trends"""
        
        healthcare_implications = {
            'mortality': {
                'increasing': 'deteriorating patient outcomes and potential quality issues',
                'decreasing': 'improving patient care and better health outcomes',
                'stable': 'consistent care quality with room for improvement'
            },
            'satisfaction': {
                'increasing': 'improving patient experience and care quality',
                'decreasing': 'declining patient experience requiring attention',
                'stable': 'consistent patient experience levels'
            },
            'cost': {
                'increasing': 'rising healthcare expenses requiring cost management',
                'decreasing': 'improving cost efficiency and resource optimization',
                'stable': 'controlled healthcare spending'
            },
            'readmission': {
                'increasing': 'potential quality issues and care coordination problems',
                'decreasing': 'improving care quality and patient outcomes',
                'stable': 'consistent care patterns'
            }
        }
        
        # Find matching healthcare domain
        for domain, implications in healthcare_implications.items():
            if domain in column.lower():
                return implications.get(direction, 'significant changes in healthcare metrics')
        
        # Default implication
        if direction == 'increasing':
            return 'upward movement in healthcare indicators'
        elif direction == 'decreasing':
            return 'downward movement in healthcare indicators'
        else:
            return 'stable healthcare performance'
    
    def _calculate_expected_range(self, series: pd.Series) -> Tuple[float, float]:
        """Calculate expected range for anomaly detection"""
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        return (lower_bound, upper_bound)
    
    def _infer_anomaly_cause(self, column: str, value: float, expected_range: Tuple[float, float]) -> str:
        """Infer potential causes for anomalies"""
        
        causes = {
            'mortality': 'quality issues, staffing problems, or patient complexity',
            'cost': 'resource inefficiency, equipment issues, or case complexity',
            'satisfaction': 'service quality issues, communication problems, or system failures',
            'wait_time': 'staffing shortages, system bottlenecks, or high demand',
            'readmission': 'discharge planning issues, care coordination problems, or patient factors'
        }
        
        for domain, cause in causes.items():
            if domain in column.lower():
                return cause
        
        return 'data quality issues, system changes, or operational factors'
    
    def _generate_correlation_implication(self, var1: str, var2: str, correlation: float) -> str:
        """Generate implications for discovered correlations"""
        
        # Healthcare-specific correlation implications
        if 'satisfaction' in var1.lower() or 'satisfaction' in var2.lower():
            if correlation > 0:
                return 'higher satisfaction is associated with better outcomes'
            else:
                return 'satisfaction may be inversely related to other factors'
        
        if 'cost' in var1.lower() or 'cost' in var2.lower():
            if correlation > 0:
                return 'cost increases are associated with other metric changes'
            else:
                return 'cost efficiency may be related to performance improvements'
        
        # General implication
        if correlation > 0:
            return 'these metrics tend to move in the same direction'
        else:
            return 'these metrics show an inverse relationship'
    
    def _identify_affected_populations(self, column: str, data: pd.DataFrame) -> List[str]:
        """Identify populations affected by the insight"""
        
        # Default populations based on common healthcare demographics
        populations = ['All patients']
        
        # Check if data has demographic columns
        demographic_columns = ['age_group', 'gender', 'race_ethnicity', 'insurance_type', 'income_level']
        
        for demo_col in demographic_columns:
            if demo_col in data.columns:
                unique_values = data[demo_col].unique()
                populations.extend([f"{demo_col.replace('_', ' ').title()}: {val}" for val in unique_values[:3]])
        
        return populations[:5]  # Limit to 5 populations
    
    def _generate_trend_recommendations(self, column: str, trend_analysis: TrendAnalysis) -> List[str]:
        """Generate recommendations based on trend analysis"""
        
        recommendations = []
        
        if trend_analysis.trend_direction == 'increasing':
            if 'cost' in column.lower() or 'mortality' in column.lower():
                recommendations.extend([
                    "Investigate root causes of the increase",
                    "Implement cost control or quality improvement measures",
                    "Monitor closely for continued escalation"
                ])
            else:
                recommendations.extend([
                    "Continue current positive practices",
                    "Identify factors driving improvement",
                    "Scale successful interventions"
                ])
        
        elif trend_analysis.trend_direction == 'decreasing':
            if 'satisfaction' in column.lower() or 'quality' in column.lower():
                recommendations.extend([
                    "Address factors causing decline",
                    "Implement improvement initiatives",
                    "Engage stakeholders for feedback"
                ])
            else:
                recommendations.extend([
                    "Maintain current improvement trajectory",
                    "Document successful practices",
                    "Continue monitoring progress"
                ])
        
        else:  # stable
            recommendations.extend([
                "Assess if current levels are optimal",
                "Consider initiatives for improvement",
                "Maintain current monitoring practices"
            ])
        
        return recommendations
    
    def _assess_trend_impact(self, column: str, trend_analysis: TrendAnalysis) -> str:
        """Assess the impact of identified trends"""
        
        impact_levels = {
            'high': ['mortality', 'safety', 'critical'],
            'medium': ['satisfaction', 'cost', 'efficiency'],
            'low': ['administrative', 'minor', 'secondary']
        }
        
        for level, keywords in impact_levels.items():
            if any(keyword in column.lower() for keyword in keywords):
                return f"{level.title()} impact on healthcare operations and outcomes"
        
        return "Moderate impact on healthcare metrics"
    
    def _calculate_trend_urgency(self, trend_analysis: TrendAnalysis) -> int:
        """Calculate urgency level for trends (1-5 scale)"""
        
        urgency = 1
        
        # Increase urgency based on trend strength
        if trend_analysis.trend_strength > 0.8:
            urgency += 2
        elif trend_analysis.trend_strength > 0.6:
            urgency += 1
        
        # Increase urgency for significant trends
        if trend_analysis.trend_significance < 0.01:
            urgency += 1
        
        # Increase urgency if there are change points
        if len(trend_analysis.change_points) > 0:
            urgency += 1
        
        return min(urgency, 5)
    
    def _generate_anomaly_recommendations(self, column: str, value: float) -> List[str]:
        """Generate recommendations for anomalies"""
        
        return [
            f"Investigate the cause of unusual {column.replace('_', ' ')} values",
            "Verify data quality and collection methods",
            "Consider if operational changes contributed to the anomaly",
            "Monitor for recurring patterns",
            "Implement corrective measures if necessary"
        ]
    
    def _assess_anomaly_impact(self, column: str, value: float) -> str:
        """Assess impact of anomalies"""
        
        critical_metrics = ['mortality', 'safety', 'infection', 'critical']
        
        if any(metric in column.lower() for metric in critical_metrics):
            return "High impact - requires immediate investigation and response"
        else:
            return "Moderate impact - should be monitored and addressed"
    
    def _calculate_anomaly_urgency(self, anomaly_score: float) -> int:
        """Calculate urgency for anomalies"""
        
        # More negative scores indicate stronger anomalies
        abs_score = abs(anomaly_score)
        
        if abs_score > 0.5:
            return 5
        elif abs_score > 0.3:
            return 4
        elif abs_score > 0.1:
            return 3
        else:
            return 2
    
    def _generate_correlation_recommendations(self, corr_info: Dict) -> List[str]:
        """Generate recommendations for correlations"""
        
        return [
            f"Investigate the relationship between {corr_info['var1']} and {corr_info['var2']}",
            "Consider if this correlation indicates causal relationships",
            "Use this insight for predictive modeling",
            "Monitor both variables together",
            "Leverage correlation for operational improvements"
        ]
    
    def _assess_correlation_impact(self, corr_info: Dict) -> str:
        """Assess impact of correlations"""
        
        if abs(corr_info['correlation']) > 0.8:
            return "High impact - strong relationship with significant implications"
        elif abs(corr_info['correlation']) > 0.6:
            return "Moderate impact - meaningful relationship for consideration"
        else:
            return "Low to moderate impact - relationship worth monitoring"
    
    def _calculate_correlation_urgency(self, corr_info: Dict) -> int:
        """Calculate urgency for correlations"""
        
        # Higher correlations get higher urgency
        abs_corr = abs(corr_info['correlation'])
        
        if abs_corr > 0.8:
            return 4
        elif abs_corr > 0.7:
            return 3
        else:
            return 2
    
    def _identify_affected_populations_correlation(self, corr_info: Dict, data: pd.DataFrame) -> List[str]:
        """Identify populations affected by correlations"""
        
        # Combine populations from both variables
        pop1 = self._identify_affected_populations(corr_info['var1'], data)
        pop2 = self._identify_affected_populations(corr_info['var2'], data)
        
        # Return unique populations
        return list(set(pop1 + pop2))[:5]
    
    def _generate_risk_description(self, column: str, risk_score: float, avg_value: float) -> str:
        """Generate risk descriptions"""
        
        risk_descriptions = {
            'mortality': f'{risk_score*100:.1f}% of cases exceed acceptable mortality thresholds',
            'readmission': f'{risk_score*100:.1f}% readmission rate indicates care quality concerns',
            'infection': f'{risk_score*100:.1f}% infection rate suggests infection control issues',
            'satisfaction': f'{risk_score*100:.1f}% of patients report low satisfaction scores',
            'cost': f'{risk_score*100:.1f}% of cases exceed budget parameters'
        }
        
        for domain, description in risk_descriptions.items():
            if domain in column.lower():
                return description
        
        return f'{risk_score*100:.1f}% of cases exceed normal parameters'
    
    def _generate_risk_recommendations(self, column: str, risk_level: str) -> str:
        """Generate risk-specific recommendations"""
        
        recommendations = {
            'high_risk': {
                'mortality': 'immediate quality review, clinical intervention protocols',
                'cost': 'urgent cost control measures, resource reallocation',
                'satisfaction': 'immediate service recovery, staff training',
                'default': 'immediate intervention, root cause analysis'
            },
            'moderate_risk': {
                'mortality': 'enhanced monitoring, quality improvement initiatives',
                'cost': 'budget review, efficiency improvements',
                'satisfaction': 'service improvement programs, feedback systems',
                'default': 'monitoring enhancement, preventive measures'
            },
            'low_risk': {
                'default': 'continued monitoring, maintain current practices'
            }
        }
        
        risk_recs = recommendations.get(risk_level, recommendations['low_risk'])
        
        for domain in risk_recs.keys():
            if domain in column.lower():
                return risk_recs[domain]
        
        return risk_recs['default']
    
    def _assess_risk_impact(self, column: str, risk_score: float) -> str:
        """Assess risk impact"""
        
        if risk_score > 0.7:
            return "Critical impact - immediate action required to prevent adverse outcomes"
        elif risk_score > 0.3:
            return "Significant impact - proactive measures needed to mitigate risks"
        else:
            return "Manageable impact - continue monitoring and maintain standards"
    
    def _recognize_patterns(self, data: pd.DataFrame, target_columns: List[str]) -> List[Insight]:
        """Recognize complex patterns in healthcare data"""
        
        insights = []
        
        # Pattern recognition using clustering
        numeric_data = data[target_columns].select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) < 2 or len(numeric_data) < 10:
            return insights
        
        # Fill missing values
        numeric_data_filled = numeric_data.fillna(numeric_data.median())
        
        # Scale data
        scaled_data = self.scaler.fit_transform(numeric_data_filled)
        
        # Perform clustering to identify patterns
        n_clusters = min(5, len(numeric_data) // 20)  # Reasonable number of clusters
        
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters for patterns
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_data = numeric_data_filled[cluster_mask]
                
                if len(cluster_data) > 5:  # Sufficient data in cluster
                    
                    # Identify cluster characteristics
                    cluster_profile = {}
                    for col in numeric_data.columns:
                        cluster_mean = cluster_data[col].mean()
                        overall_mean = numeric_data_filled[col].mean()
                        
                        # Calculate relative difference
                        if overall_mean != 0:
                            relative_diff = (cluster_mean - overall_mean) / overall_mean
                            if abs(relative_diff) > 0.2:  # 20% difference threshold
                                cluster_profile[col] = {
                                    'value': cluster_mean,
                                    'difference': relative_diff,
                                    'direction': 'higher' if relative_diff > 0 else 'lower'
                                }
                    
                    if cluster_profile:  # If cluster has distinctive characteristics
                        
                        # Generate pattern description
                        pattern_description = self._generate_pattern_description(cluster_profile, cluster_id)
                        
                        # Calculate confidence based on cluster separation
                        cluster_size_ratio = len(cluster_data) / len(numeric_data_filled)
                        confidence_score = min(cluster_size_ratio * 2, 1.0)  # Larger clusters = higher confidence
                        confidence_level = self._score_to_confidence_level(confidence_score)
                        
                        insight = Insight(
                            type=InsightType.PATTERN_RECOGNITION,
                            title=f"Pattern Identified: Cluster {cluster_id + 1}",
                            description=f"Distinct pattern found in {len(cluster_data)} cases ({cluster_size_ratio*100:.1f}% of data)",
                            natural_language_explanation=pattern_description,
                            confidence=confidence_level,
                            confidence_score=confidence_score,
                            supporting_evidence=[
                                f"Cluster size: {len(cluster_data)} cases",
                                f"Percentage of total: {cluster_size_ratio*100:.1f}%",
                                f"Distinctive features: {len(cluster_profile)}"
                            ],
                            data_points={
                                'cluster_id': cluster_id,
                                'cluster_size': len(cluster_data),
                                'cluster_profile': cluster_profile
                            },
                            recommendations=self._generate_pattern_recommendations(cluster_profile),
                            impact_assessment=self._assess_pattern_impact(cluster_profile),
                            urgency_level=self._calculate_pattern_urgency(cluster_profile),
                            affected_populations=[f"Patient group {cluster_id + 1}"]
                        )
                        
                        insights.append(insight)
        
        return insights
    
    def _generate_pattern_description(self, cluster_profile: Dict, cluster_id: int) -> str:
        """Generate natural language description of identified patterns"""
        
        description_parts = []
        
        # Describe the most significant characteristics
        sorted_features = sorted(cluster_profile.items(), 
                               key=lambda x: abs(x[1]['difference']), 
                               reverse=True)
        
        for feature, info in sorted_features[:3]:  # Top 3 characteristics
            feature_name = feature.replace('_', ' ').title()
            direction = info['direction']
            percentage = abs(info['difference']) * 100
            
            description_parts.append(f"{feature_name} is {percentage:.1f}% {direction} than average")
        
        if description_parts:
            main_description = f"This pattern represents a distinct group with {', '.join(description_parts[:2])}"
            if len(description_parts) > 2:
                main_description += f", and {description_parts[2]}"
            
            main_description += f". This suggests a specific patient or operational profile that may require targeted interventions or specialized care approaches."
        else:
            main_description = f"A distinct pattern has been identified in the data representing a unique subset of cases."
        
        return main_description
    
    def _generate_pattern_recommendations(self, cluster_profile: Dict) -> List[str]:
        """Generate recommendations for identified patterns"""
        
        recommendations = [
            "Develop targeted interventions for this specific patient group",
            "Analyze the characteristics that define this pattern",
            "Consider specialized care protocols for this population",
            "Monitor this group separately for better outcomes tracking",
            "Investigate if this pattern represents an opportunity for improvement"
        ]
        
        return recommendations
    
    def _assess_pattern_impact(self, cluster_profile: Dict) -> str:
        """Assess the impact of identified patterns"""
        
        # Check if pattern involves critical metrics
        critical_metrics = ['mortality', 'safety', 'infection', 'readmission']
        
        has_critical = any(metric in feature.lower() 
                          for feature in cluster_profile.keys() 
                          for metric in critical_metrics)
        
        if has_critical:
            return "High impact - pattern involves critical healthcare metrics requiring attention"
        else:
            return "Moderate impact - pattern provides insights for operational improvements"
    
    def _calculate_pattern_urgency(self, cluster_profile: Dict) -> int:
        """Calculate urgency for identified patterns"""
        
        # Base urgency
        urgency = 2
        
        # Increase urgency for large deviations
        max_deviation = max(abs(info['difference']) for info in cluster_profile.values())
        
        if max_deviation > 0.5:  # 50% deviation
            urgency += 2
        elif max_deviation > 0.3:  # 30% deviation
            urgency += 1
        
        # Check for critical metrics
        critical_metrics = ['mortality', 'safety', 'infection']
        has_critical = any(metric in feature.lower() 
                          for feature in cluster_profile.keys() 
                          for metric in critical_metrics)
        
        if has_critical:
            urgency += 1
        
        return min(urgency, 5)
    
    def _generate_forecasts(self, data: pd.DataFrame, 
                          target_columns: List[str], 
                          time_column: str) -> List[Insight]:
        """Generate predictive forecasts using AI models"""
        
        insights = []
        
        # Sort by time
        data_sorted = data.sort_values(time_column)
        
        for column in target_columns:
            if column in data_sorted.columns and data_sorted[column].notna().sum() > 10:
                
                # Prepare time series data
                ts_data = data_sorted[[time_column, column]].dropna()
                
                if len(ts_data) < 10:
                    continue
                
                # Simple forecast using trend analysis
                values = ts_data[column].values
                forecast_values, confidence_intervals = self._generate_simple_forecast(values, periods=3)
                
                if forecast_values:
                    # Determine forecast direction
                    current_value = values[-1]
                    future_value = forecast_values[0]
                    
                    if future_value > current_value * 1.05:
                        direction = "increase"
                    elif future_value < current_value * 0.95:
                        direction = "decrease"
                    else:
                        direction = "remain stable"
                    
                    # Generate implication
                    implication = self._generate_forecast_implication(column, direction, future_value)
                    
                    # Create explanation
                    template = self.language_templates[InsightType.PREDICTIVE_FORECAST]['forecast']
                    explanation = template.format(
                        metric=column.replace('_', ' ').title(),
                        direction=direction,
                        predicted_value=f"{future_value:.2f}",
                        time_horizon="next period",
                        lower_bound=f"{confidence_intervals[0][0]:.2f}",
                        upper_bound=f"{confidence_intervals[0][1]:.2f}",
                        implication=implication
                    )
                    
                    # Calculate confidence
                    # Base confidence on trend strength and data quality
                    trend_analysis = self._analyze_trend(values)
                    confidence_score = min(trend_analysis.trend_strength * 0.8, 1.0)
                    confidence_level = self._score_to_confidence_level(confidence_score)
                    
                    insight = Insight(
                        type=InsightType.PREDICTIVE_FORECAST,
                        title=f"Forecast: {column.replace('_', ' ').title()}",
                        description=f"Predicted to {direction} to {future_value:.2f}",
                        natural_language_explanation=explanation,
                        confidence=confidence_level,
                        confidence_score=confidence_score,
                        supporting_evidence=[
                            f"Current value: {current_value:.2f}",
                            f"Predicted value: {future_value:.2f}",
                            f"Confidence interval: [{confidence_intervals[0][0]:.2f}, {confidence_intervals[0][1]:.2f}]",
                            f"Based on {len(values)} historical data points"
                        ],
                        data_points={
                            'current_value': current_value,
                            'predicted_value': future_value,
                            'confidence_interval': confidence_intervals[0],
                            'forecast_direction': direction
                        },
                        recommendations=self._generate_forecast_recommendations(column, direction, future_value),
                        impact_assessment=self._assess_forecast_impact(column, direction, future_value),
                        urgency_level=self._calculate_forecast_urgency(column, direction, future_value),
                        affected_populations=self._identify_affected_populations(column, data)
                    )
                    
                    insights.append(insight)
        
        return insights
    
    def _generate_forecast_implication(self, column: str, direction: str, value: float) -> str:
        """Generate implications for forecasts"""
        
        implications = {
            'mortality': {
                'increase': 'potential deterioration in patient outcomes requiring intervention',
                'decrease': 'expected improvement in patient care and safety',
                'remain stable': 'consistent mortality patterns with current care levels'
            },
            'cost': {
                'increase': 'rising healthcare expenses requiring budget planning',
                'decrease': 'cost savings and improved efficiency',
                'remain stable': 'controlled healthcare spending'
            },
            'satisfaction': {
                'increase': 'improving patient experience and care quality',
                'decrease': 'declining patient satisfaction requiring attention',
                'remain stable': 'consistent patient experience levels'
            }
        }
        
        for domain, domain_implications in implications.items():
            if domain in column.lower():
                return domain_implications.get(direction, 'changes in healthcare metrics')
        
        return f'expected {direction} in healthcare performance indicators'
    
    def _generate_forecast_recommendations(self, column: str, direction: str, value: float) -> List[str]:
        """Generate recommendations for forecasts"""
        
        if direction == "increase":
            if any(term in column.lower() for term in ['cost', 'mortality', 'readmission']):
                return [
                    "Prepare intervention strategies to prevent negative outcomes",
                    "Allocate additional resources if needed",
                    "Monitor closely for early warning signs",
                    "Review current protocols and procedures"
                ]
            else:
                return [
                    "Prepare to capitalize on positive trends",
                    "Scale successful practices",
                    "Maintain current improvement strategies"
                ]
        
        elif direction == "decrease":
            if any(term in column.lower() for term in ['satisfaction', 'quality', 'efficiency']):
                return [
                    "Implement improvement initiatives immediately",
                    "Address root causes of decline",
                    "Engage stakeholders for feedback and solutions"
                ]
            else:
                return [
                    "Continue current successful strategies",
                    "Document best practices",
                    "Maintain improvement momentum"
                ]
        
        else:  # remain stable
            return [
                "Assess if current levels are optimal",
                "Consider initiatives for further improvement",
                "Maintain monitoring and current practices"
            ]
    
    def _assess_forecast_impact(self, column: str, direction: str, value: float) -> str:
        """Assess impact of forecasts"""
        
        critical_metrics = ['mortality', 'safety', 'infection', 'readmission']
        
        if any(metric in column.lower() for metric in critical_metrics):
            if direction == "increase":
                return "High impact - predicted increase in critical metric requires immediate planning"
            else:
                return "Positive impact - predicted improvement in critical healthcare metric"
        else:
            return f"Moderate impact - predicted {direction} will affect operational planning"
    
    def _calculate_forecast_urgency(self, column: str, direction: str, value: float) -> int:
        """Calculate urgency for forecasts"""
        
        urgency = 2  # Base urgency
        
        # Increase urgency for critical metrics
        critical_metrics = ['mortality', 'safety', 'infection']
        if any(metric in column.lower() for metric in critical_metrics):
            urgency += 2
        
        # Increase urgency for negative trends in positive metrics
        positive_metrics = ['satisfaction', 'quality', 'efficiency']
        if any(metric in column.lower() for metric in positive_metrics) and direction == "decrease":
            urgency += 1
        
        # Increase urgency for positive trends in negative metrics
        negative_metrics = ['cost', 'mortality', 'readmission', 'infection']
        if any(metric in column.lower() for metric in negative_metrics) and direction == "increase":
            urgency += 1
        
        return min(urgency, 5)

# Example usage
if __name__ == "__main__":
    # Create sample healthcare data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'mortality_rate': np.random.beta(2, 50, n_samples),
        'satisfaction_score': np.random.normal(7.5, 1.5, n_samples),
        'readmission_rate': np.random.beta(3, 20, n_samples),
        'cost_per_patient': np.random.lognormal(8.5, 0.3, n_samples),
        'wait_time_minutes': np.random.gamma(2, 15, n_samples),
        'staff_satisfaction': np.random.normal(6.8, 1.2, n_samples)
    })
    
    # Initialize AI insights engine
    ai_engine = AIHealthcareInsightsEngine()
    
    # Generate comprehensive insights
    insights = ai_engine.analyze_comprehensive_insights(
        sample_data, 
        target_columns=['mortality_rate', 'satisfaction_score', 'readmission_rate', 'cost_per_patient'],
        time_column='date'
    )
    
    print(f"Generated {len(insights)} AI-powered insights:")
    
    for i, insight in enumerate(insights[:5], 1):  # Show top 5 insights
        print(f"\n{i}. {insight.title}")
        print(f"   Type: {insight.type.value}")
        print(f"   Confidence: {insight.confidence.value} ({insight.confidence_score:.2f})")
        print(f"   Urgency: {insight.urgency_level}/5")
        print(f"   Description: {insight.natural_language_explanation[:200]}...")
        print(f"   Recommendations: {', '.join(insight.recommendations[:2])}")
    
    print("\nAI Healthcare Insights Engine Features:")
    print("- Automated trend detection with statistical significance")
    print("- Anomaly identification using machine learning")
    print("- Pattern recognition through clustering analysis")
    print("- Correlation discovery with healthcare context")
    print("- Risk assessment with urgency scoring")
    print("- Predictive forecasting with confidence intervals")
    print("- Natural language explanations for all insights")
    print("- Contextual recommendations and impact assessment")