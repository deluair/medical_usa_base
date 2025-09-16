"""
Advanced Monte Carlo Simulation Engine for Healthcare Analytics
Sophisticated modeling with nuanced scenarios, shades of uncertainty, and complex interdependencies
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ScenarioType(Enum):
    """Types of healthcare scenarios to simulate"""
    PANDEMIC_IMPACT = "pandemic_impact"
    POLICY_CHANGE = "policy_change"
    MARKET_DISRUPTION = "market_disruption"
    DEMOGRAPHIC_SHIFT = "demographic_shift"
    TECHNOLOGY_ADOPTION = "technology_adoption"
    ECONOMIC_SHOCK = "economic_shock"
    REGULATORY_CHANGE = "regulatory_change"

class UncertaintyLevel(Enum):
    """Levels of uncertainty in simulations"""
    LOW = "low"          # ±5% variation
    MODERATE = "moderate" # ±15% variation
    HIGH = "high"        # ±30% variation
    EXTREME = "extreme"  # ±50% variation

@dataclass
class SimulationParameter:
    """Parameter for Monte Carlo simulation with nuanced distributions"""
    name: str
    base_value: float
    distribution: str = "normal"  # normal, lognormal, beta, gamma, triangular
    uncertainty_level: UncertaintyLevel = UncertaintyLevel.MODERATE
    correlation_matrix: Optional[Dict[str, float]] = None
    seasonal_factor: Optional[float] = None
    trend_factor: Optional[float] = None
    shock_probability: float = 0.05  # Probability of extreme events
    shock_magnitude: float = 2.0     # Multiplier for extreme events
    
    def __post_init__(self):
        if self.correlation_matrix is None:
            self.correlation_matrix = {}

@dataclass
class ScenarioConfig:
    """Configuration for a specific healthcare scenario"""
    scenario_type: ScenarioType
    name: str
    description: str
    parameters: List[SimulationParameter]
    time_horizon: int = 60  # months
    simulation_runs: int = 10000
    confidence_intervals: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    interdependencies: Dict[str, Dict[str, float]] = field(default_factory=dict)

class AdvancedMonteCarloEngine:
    """Advanced Monte Carlo simulation engine with sophisticated modeling"""
    
    def __init__(self):
        self.scenarios = {}
        self.results_cache = {}
        self.correlation_matrices = {}
        
    def add_scenario(self, config: ScenarioConfig):
        """Add a new scenario configuration"""
        self.scenarios[config.name] = config
        logger.info(f"Added scenario: {config.name} ({config.scenario_type.value})")
    
    def generate_correlated_samples(self, parameters: List[SimulationParameter], 
                                  n_samples: int) -> np.ndarray:
        """Generate correlated samples using Cholesky decomposition"""
        n_params = len(parameters)
        
        # Build correlation matrix
        correlation_matrix = np.eye(n_params)
        param_names = [p.name for p in parameters]
        
        for i, param_i in enumerate(parameters):
            for j, param_j in enumerate(parameters):
                if i != j and param_j.name in param_i.correlation_matrix:
                    correlation_matrix[i, j] = param_i.correlation_matrix[param_j.name]
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.001)  # Regularization
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # Fallback to identity if decomposition fails
            L = np.eye(n_params)
            logger.warning("Correlation matrix not positive definite, using identity")
        
        # Generate independent samples
        independent_samples = np.random.standard_normal((n_samples, n_params))
        
        # Apply correlation
        correlated_samples = independent_samples @ L.T
        
        # Transform to desired distributions
        transformed_samples = np.zeros_like(correlated_samples)
        
        for i, param in enumerate(parameters):
            samples = correlated_samples[:, i]
            transformed_samples[:, i] = self._transform_to_distribution(samples, param)
        
        return transformed_samples
    
    def _transform_to_distribution(self, standard_normal_samples: np.ndarray, 
                                 param: SimulationParameter) -> np.ndarray:
        """Transform standard normal samples to desired distribution"""
        base_value = param.base_value
        
        # Get uncertainty range based on level
        uncertainty_ranges = {
            UncertaintyLevel.LOW: 0.05,
            UncertaintyLevel.MODERATE: 0.15,
            UncertaintyLevel.HIGH: 0.30,
            UncertaintyLevel.EXTREME: 0.50
        }
        
        uncertainty = uncertainty_ranges[param.uncertainty_level]
        
        if param.distribution == "normal":
            std_dev = base_value * uncertainty
            samples = base_value + standard_normal_samples * std_dev
            
        elif param.distribution == "lognormal":
            # For lognormal, we want the median to be base_value
            sigma = uncertainty
            mu = np.log(base_value) - 0.5 * sigma**2
            samples = np.exp(mu + sigma * standard_normal_samples)
            
        elif param.distribution == "beta":
            # Beta distribution scaled to [base_value*(1-uncertainty), base_value*(1+uncertainty)]
            alpha, beta = 2, 2  # Symmetric beta
            uniform_samples = stats.norm.cdf(standard_normal_samples)
            beta_samples = stats.beta.ppf(uniform_samples, alpha, beta)
            min_val = base_value * (1 - uncertainty)
            max_val = base_value * (1 + uncertainty)
            samples = min_val + beta_samples * (max_val - min_val)
            
        elif param.distribution == "gamma":
            # Gamma distribution with mean = base_value
            cv = uncertainty  # Coefficient of variation
            shape = 1 / cv**2
            scale = base_value / shape
            uniform_samples = stats.norm.cdf(standard_normal_samples)
            samples = stats.gamma.ppf(uniform_samples, shape, scale=scale)
            
        elif param.distribution == "triangular":
            # Triangular distribution
            min_val = base_value * (1 - uncertainty)
            max_val = base_value * (1 + uncertainty)
            mode = base_value
            uniform_samples = stats.norm.cdf(standard_normal_samples)
            samples = stats.triang.ppf(uniform_samples, 
                                     (mode - min_val) / (max_val - min_val),
                                     loc=min_val, scale=max_val - min_val)
        else:
            # Default to normal
            std_dev = base_value * uncertainty
            samples = base_value + standard_normal_samples * std_dev
        
        # Apply shock events
        shock_mask = np.random.random(len(samples)) < param.shock_probability
        shock_multiplier = np.where(
            np.random.random(len(samples)) < 0.5, 
            param.shock_magnitude, 
            1.0 / param.shock_magnitude
        )
        samples[shock_mask] *= shock_multiplier[shock_mask]
        
        return np.maximum(samples, 0)  # Ensure non-negative values
    
    def run_pandemic_simulation(self, severity: str = "moderate") -> Dict[str, Any]:
        """Run pandemic impact simulation with nuanced modeling"""
        
        # Define pandemic parameters based on severity
        severity_configs = {
            "mild": {
                "infection_rate": 0.15,
                "hospitalization_rate": 0.02,
                "icu_rate": 0.005,
                "mortality_rate": 0.001,
                "economic_impact": 0.10,
                "healthcare_capacity_strain": 0.20
            },
            "moderate": {
                "infection_rate": 0.35,
                "hospitalization_rate": 0.05,
                "icu_rate": 0.015,
                "mortality_rate": 0.003,
                "economic_impact": 0.25,
                "healthcare_capacity_strain": 0.50
            },
            "severe": {
                "infection_rate": 0.60,
                "hospitalization_rate": 0.10,
                "icu_rate": 0.03,
                "mortality_rate": 0.008,
                "economic_impact": 0.45,
                "healthcare_capacity_strain": 0.80
            }
        }
        
        config = severity_configs.get(severity, severity_configs["moderate"])
        
        # Create simulation parameters
        parameters = [
            SimulationParameter(
                name="infection_rate",
                base_value=config["infection_rate"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"hospitalization_rate": 0.7, "economic_impact": 0.6}
            ),
            SimulationParameter(
                name="hospitalization_rate",
                base_value=config["hospitalization_rate"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"icu_rate": 0.8, "healthcare_costs": 0.9}
            ),
            SimulationParameter(
                name="icu_rate",
                base_value=config["icu_rate"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.EXTREME,
                correlation_matrix={"mortality_rate": 0.7, "healthcare_capacity_strain": 0.8}
            ),
            SimulationParameter(
                name="mortality_rate",
                base_value=config["mortality_rate"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.EXTREME,
                shock_probability=0.1,
                shock_magnitude=3.0
            ),
            SimulationParameter(
                name="economic_impact",
                base_value=config["economic_impact"],
                distribution="gamma",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"healthcare_spending": -0.5}
            ),
            SimulationParameter(
                name="healthcare_capacity_strain",
                base_value=config["healthcare_capacity_strain"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"quality_of_care": -0.6}
            )
        ]
        
        # Run simulation
        n_simulations = 10000
        samples = self.generate_correlated_samples(parameters, n_simulations)
        
        # Calculate derived metrics
        results = {}
        param_names = [p.name for p in parameters]
        
        for i, name in enumerate(param_names):
            results[name] = samples[:, i]
        
        # Calculate complex derived metrics
        population = 330_000_000  # US population
        
        results["total_infections"] = results["infection_rate"] * population
        results["total_hospitalizations"] = results["total_infections"] * results["hospitalization_rate"]
        results["total_icu_admissions"] = results["total_hospitalizations"] * results["icu_rate"]
        results["total_deaths"] = results["total_infections"] * results["mortality_rate"]
        
        # Healthcare system impact
        results["healthcare_cost_increase"] = (
            results["total_hospitalizations"] * 15000 +  # Average hospitalization cost
            results["total_icu_admissions"] * 50000      # Average ICU cost
        ) * (1 + results["healthcare_capacity_strain"])
        
        # Economic impact
        gdp = 25_000_000_000_000  # US GDP
        results["gdp_loss"] = gdp * results["economic_impact"]
        
        # Quality metrics with nuanced relationships
        results["quality_of_care_index"] = np.maximum(
            0.1, 1.0 - results["healthcare_capacity_strain"] * 1.2
        )
        
        results["healthcare_worker_burnout"] = np.minimum(
            1.0, results["healthcare_capacity_strain"] * 1.5
        )
        
        # Calculate confidence intervals and statistics
        summary_stats = {}
        for metric, values in results.items():
            summary_stats[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "percentiles": {
                    "5th": np.percentile(values, 5),
                    "25th": np.percentile(values, 25),
                    "75th": np.percentile(values, 75),
                    "95th": np.percentile(values, 95)
                }
            }
        
        return {
            "scenario": f"pandemic_{severity}",
            "raw_results": results,
            "summary_statistics": summary_stats,
            "simulation_parameters": {p.name: p.base_value for p in parameters},
            "n_simulations": n_simulations,
            "correlations": self._calculate_correlations(results)
        }
    
    def run_policy_impact_simulation(self, policy_type: str) -> Dict[str, Any]:
        """Simulate healthcare policy impact with nuanced effects"""
        
        policy_configs = {
            "universal_healthcare": {
                "coverage_increase": 0.15,
                "cost_per_capita_change": 0.20,
                "administrative_efficiency": 0.25,
                "wait_times_change": 0.10,
                "preventive_care_increase": 0.30,
                "implementation_cost": 2_000_000_000_000
            },
            "medicare_for_all": {
                "coverage_increase": 0.25,
                "cost_per_capita_change": -0.10,
                "administrative_efficiency": 0.40,
                "wait_times_change": 0.15,
                "preventive_care_increase": 0.40,
                "implementation_cost": 3_500_000_000_000
            },
            "price_transparency": {
                "coverage_increase": 0.02,
                "cost_per_capita_change": -0.08,
                "administrative_efficiency": 0.15,
                "wait_times_change": -0.05,
                "preventive_care_increase": 0.10,
                "implementation_cost": 50_000_000_000
            },
            "drug_price_regulation": {
                "coverage_increase": 0.05,
                "cost_per_capita_change": -0.15,
                "administrative_efficiency": 0.05,
                "wait_times_change": 0.02,
                "preventive_care_increase": 0.08,
                "implementation_cost": 100_000_000_000
            }
        }
        
        config = policy_configs.get(policy_type, policy_configs["price_transparency"])
        
        # Create nuanced parameters with complex interdependencies
        parameters = [
            SimulationParameter(
                name="coverage_increase",
                base_value=config["coverage_increase"],
                distribution="beta",
                uncertainty_level=UncertaintyLevel.MODERATE,
                correlation_matrix={"preventive_care_increase": 0.6, "cost_change": 0.4}
            ),
            SimulationParameter(
                name="cost_per_capita_change",
                base_value=config["cost_per_capita_change"],
                distribution="normal",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"administrative_efficiency": -0.5}
            ),
            SimulationParameter(
                name="administrative_efficiency",
                base_value=config["administrative_efficiency"],
                distribution="gamma",
                uncertainty_level=UncertaintyLevel.MODERATE,
                correlation_matrix={"wait_times_change": -0.3}
            ),
            SimulationParameter(
                name="wait_times_change",
                base_value=config["wait_times_change"],
                distribution="normal",
                uncertainty_level=UncertaintyLevel.HIGH,
                correlation_matrix={"quality_impact": -0.4}
            ),
            SimulationParameter(
                name="preventive_care_increase",
                base_value=config["preventive_care_increase"],
                distribution="lognormal",
                uncertainty_level=UncertaintyLevel.MODERATE,
                correlation_matrix={"long_term_savings": 0.7}
            )
        ]
        
        # Run simulation
        n_simulations = 10000
        samples = self.generate_correlated_samples(parameters, n_simulations)
        
        results = {}
        param_names = [p.name for p in parameters]
        
        for i, name in enumerate(param_names):
            results[name] = samples[:, i]
        
        # Calculate complex policy outcomes
        current_healthcare_spending = 4_300_000_000_000  # Current US healthcare spending
        population = 330_000_000
        
        # Direct impacts
        results["new_coverage_population"] = results["coverage_increase"] * population
        results["annual_cost_change"] = (
            current_healthcare_spending * results["cost_per_capita_change"]
        )
        
        # Long-term effects with nuanced modeling
        results["preventive_care_savings"] = (
            results["preventive_care_increase"] * 
            current_healthcare_spending * 0.15 *  # Preventive care is ~15% of spending
            np.random.gamma(2, 0.5, n_simulations)  # Savings multiplier with uncertainty
        )
        
        # Quality and access improvements
        results["quality_of_care_change"] = (
            results["administrative_efficiency"] * 0.3 +
            results["preventive_care_increase"] * 0.4 -
            results["wait_times_change"] * 0.3
        )
        
        # Economic multiplier effects
        results["economic_multiplier"] = np.random.gamma(1.5, 0.3, n_simulations)
        results["total_economic_impact"] = (
            results["annual_cost_change"] * results["economic_multiplier"]
        )
        
        # Implementation challenges (nuanced risk factors)
        results["implementation_success_probability"] = np.minimum(
            0.95,
            np.maximum(
                0.1,
                0.7 - results["administrative_efficiency"] * 0.2 +
                np.random.normal(0, 0.15, n_simulations)
            )
        )
        
        # Political feasibility (complex social dynamics)
        results["political_support"] = np.minimum(
            1.0,
            np.maximum(
                0.0,
                0.5 + results["quality_of_care_change"] * 0.3 -
                np.abs(results["cost_per_capita_change"]) * 0.4 +
                np.random.normal(0, 0.2, n_simulations)
            )
        )
        
        # Calculate summary statistics
        summary_stats = {}
        for metric, values in results.items():
            summary_stats[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "percentiles": {
                    "5th": np.percentile(values, 5),
                    "25th": np.percentile(values, 25),
                    "75th": np.percentile(values, 75),
                    "95th": np.percentile(values, 95)
                }
            }
        
        return {
            "scenario": f"policy_{policy_type}",
            "raw_results": results,
            "summary_statistics": summary_stats,
            "policy_parameters": config,
            "n_simulations": n_simulations,
            "correlations": self._calculate_correlations(results)
        }
    
    def _calculate_correlations(self, results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix for all result variables"""
        df = pd.DataFrame(results)
        correlation_matrix = df.corr()
        
        correlations = {}
        for col1 in correlation_matrix.columns:
            correlations[col1] = {}
            for col2 in correlation_matrix.columns:
                if col1 != col2:
                    correlations[col1][col2] = correlation_matrix.loc[col1, col2]
        
        return correlations
    
    def run_market_disruption_simulation(self, disruption_type: str) -> Dict[str, Any]:
        """Simulate market disruption scenarios with cascading effects"""
        
        disruption_configs = {
            "ai_automation": {
                "job_displacement_rate": 0.25,
                "efficiency_gain": 0.40,
                "cost_reduction": 0.30,
                "quality_improvement": 0.20,
                "adoption_speed": 0.15,
                "resistance_factor": 0.30
            },
            "telemedicine_boom": {
                "market_penetration": 0.60,
                "cost_reduction": 0.25,
                "access_improvement": 0.45,
                "quality_change": -0.05,
                "infrastructure_investment": 500_000_000_000,
                "regulatory_lag": 0.20
            },
            "biotech_breakthrough": {
                "treatment_efficacy": 0.80,
                "cost_increase": 2.50,
                "market_concentration": 0.40,
                "access_inequality": 0.35,
                "research_investment": 1_000_000_000_000,
                "time_to_market": 8.0  # years
            }
        }
        
        config = disruption_configs.get(disruption_type, disruption_configs["ai_automation"])
        
        # Create parameters with complex interdependencies
        parameters = []
        for key, value in config.items():
            uncertainty_level = UncertaintyLevel.HIGH if "rate" in key or "factor" in key else UncertaintyLevel.MODERATE
            
            parameters.append(
                SimulationParameter(
                    name=key,
                    base_value=value,
                    distribution="lognormal" if "investment" in key else "beta",
                    uncertainty_level=uncertainty_level,
                    shock_probability=0.08,
                    shock_magnitude=1.8
                )
            )
        
        # Run simulation with cascading effects
        n_simulations = 10000
        samples = self.generate_correlated_samples(parameters, n_simulations)
        
        results = {}
        param_names = [p.name for p in parameters]
        
        for i, name in enumerate(param_names):
            results[name] = samples[:, i]
        
        # Model cascading effects and system dynamics
        if disruption_type == "ai_automation":
            results["healthcare_jobs_lost"] = results["job_displacement_rate"] * 16_000_000  # Healthcare workers
            results["productivity_gain"] = results["efficiency_gain"] * results["adoption_speed"]
            results["net_cost_savings"] = (
                results["cost_reduction"] * 4_300_000_000_000 *  # Total healthcare spending
                (1 - results["resistance_factor"])
            )
            results["transition_costs"] = (
                results["healthcare_jobs_lost"] * 50000 *  # Retraining cost per worker
                results["resistance_factor"]
            )
            
        elif disruption_type == "telemedicine_boom":
            results["patients_served_remotely"] = results["market_penetration"] * 330_000_000
            results["rural_access_improvement"] = results["access_improvement"] * 0.6  # Higher impact in rural areas
            results["urban_convenience_gain"] = results["access_improvement"] * 0.4
            results["infrastructure_roi"] = (
                results["cost_reduction"] * 4_300_000_000_000 / 
                results["infrastructure_investment"]
            )
            
        elif disruption_type == "biotech_breakthrough":
            results["lives_saved_annually"] = (
                results["treatment_efficacy"] * 600000 *  # Annual deaths from target diseases
                np.random.beta(2, 3, n_simulations)  # Gradual adoption curve
            )
            results["healthcare_inequality_index"] = (
                results["access_inequality"] * results["cost_increase"]
            )
            results["market_value_created"] = (
                results["treatment_efficacy"] * 5_000_000_000_000 *  # Value of statistical life
                results["lives_saved_annually"]
            )
        
        # Universal metrics for all disruption types
        results["adoption_timeline"] = np.random.gamma(
            shape=2, scale=3, size=n_simulations
        )  # Years to full adoption
        
        results["regulatory_adaptation_lag"] = np.random.exponential(
            scale=2, size=n_simulations
        )  # Years for regulatory catch-up
        
        results["social_acceptance"] = np.random.beta(
            a=3, b=2, size=n_simulations
        )  # Public acceptance probability
        
        # Calculate summary statistics
        summary_stats = {}
        for metric, values in results.items():
            summary_stats[metric] = {
                "mean": np.mean(values),
                "median": np.median(values),
                "std": np.std(values),
                "percentiles": {
                    "5th": np.percentile(values, 5),
                    "25th": np.percentile(values, 25),
                    "75th": np.percentile(values, 75),
                    "95th": np.percentile(values, 95)
                }
            }
        
        return {
            "scenario": f"disruption_{disruption_type}",
            "raw_results": results,
            "summary_statistics": summary_stats,
            "disruption_parameters": config,
            "n_simulations": n_simulations,
            "correlations": self._calculate_correlations(results)
        }

# Example usage and testing
if __name__ == "__main__":
    engine = AdvancedMonteCarloEngine()
    
    # Run pandemic simulation
    pandemic_results = engine.run_pandemic_simulation("moderate")
    print("Pandemic Simulation Results:")
    print(f"Expected deaths: {pandemic_results['summary_statistics']['total_deaths']['mean']:,.0f}")
    print(f"Healthcare cost increase: ${pandemic_results['summary_statistics']['healthcare_cost_increase']['mean']:,.0f}")
    
    # Run policy simulation
    policy_results = engine.run_policy_impact_simulation("universal_healthcare")
    print("\nPolicy Simulation Results:")
    print(f"New coverage: {policy_results['summary_statistics']['new_coverage_population']['mean']:,.0f} people")
    print(f"Annual cost change: ${policy_results['summary_statistics']['annual_cost_change']['mean']:,.0f}")
    
    # Run market disruption simulation
    disruption_results = engine.run_market_disruption_simulation("ai_automation")
    print("\nMarket Disruption Results:")
    print(f"Jobs potentially displaced: {disruption_results['summary_statistics']['healthcare_jobs_lost']['mean']:,.0f}")
    print(f"Net cost savings: ${disruption_results['summary_statistics']['net_cost_savings']['mean']:,.0f}")