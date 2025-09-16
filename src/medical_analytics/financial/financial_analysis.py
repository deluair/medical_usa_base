"""
Advanced Financial Analysis for Healthcare Industry
Includes ROI calculations, market valuations, investment analysis, and financial projections
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class HealthcareROIAnalyzer:
    """Analyzes Return on Investment for healthcare projects and investments"""
    
    def __init__(self):
        self.discount_rate = 0.08  # Default 8% discount rate
        self.tax_rate = 0.21  # Corporate tax rate
        
    def calculate_npv(self, cash_flows: List[float], discount_rate: float = None) -> float:
        """Calculate Net Present Value of cash flows"""
        if discount_rate is None:
            discount_rate = self.discount_rate
            
        npv = 0
        for i, cash_flow in enumerate(cash_flows):
            npv += cash_flow / ((1 + discount_rate) ** i)
        
        return npv
    
    def calculate_irr(self, cash_flows: List[float], max_iterations: int = 1000) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        def npv_function(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        def npv_derivative(rate):
            return sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows))
        
        # Initial guess
        rate = 0.1
        
        for _ in range(max_iterations):
            npv_val = npv_function(rate)
            npv_deriv = npv_derivative(rate)
            
            if abs(npv_val) < 1e-6:
                return rate
            
            if npv_deriv == 0:
                break
                
            rate = rate - npv_val / npv_deriv
            
            if rate < -0.99:  # Prevent negative rates below -99%
                rate = -0.99
        
        return rate
    
    def analyze_hospital_investment(self, 
                                  initial_investment: float,
                                  annual_revenue: float,
                                  annual_costs: float,
                                  years: int = 10,
                                  revenue_growth: float = 0.03,
                                  cost_inflation: float = 0.025) -> Dict[str, any]:
        """Analyze ROI for hospital investment"""
        
        logger.info(f"Analyzing hospital investment ROI over {years} years...")
        
        # Generate cash flows
        cash_flows = [-initial_investment]  # Initial investment (negative)
        
        current_revenue = annual_revenue
        current_costs = annual_costs
        
        yearly_analysis = []
        
        for year in range(1, years + 1):
            # Apply growth rates
            current_revenue *= (1 + revenue_growth)
            current_costs *= (1 + cost_inflation)
            
            # Calculate annual cash flow
            ebitda = current_revenue - current_costs
            depreciation = initial_investment / years  # Straight-line depreciation
            ebit = ebitda - depreciation
            taxes = max(0, ebit * self.tax_rate)
            net_income = ebit - taxes
            annual_cash_flow = net_income + depreciation  # Add back depreciation
            
            cash_flows.append(annual_cash_flow)
            
            yearly_analysis.append({
                'year': year,
                'revenue': current_revenue,
                'costs': current_costs,
                'ebitda': ebitda,
                'ebit': ebit,
                'taxes': taxes,
                'net_income': net_income,
                'cash_flow': annual_cash_flow,
                'cumulative_cash_flow': sum(cash_flows)
            })
        
        # Calculate financial metrics
        npv = self.calculate_npv(cash_flows)
        irr = self.calculate_irr(cash_flows)
        payback_period = self._calculate_payback_period(cash_flows)
        
        # Calculate profitability ratios
        total_cash_flows = sum(cash_flows[1:])  # Exclude initial investment
        roi_percentage = (total_cash_flows / initial_investment) * 100
        
        return {
            'investment_summary': {
                'initial_investment': initial_investment,
                'npv': npv,
                'irr': irr * 100,  # Convert to percentage
                'payback_period_years': payback_period,
                'total_roi_percentage': roi_percentage,
                'is_profitable': npv > 0
            },
            'yearly_projections': yearly_analysis,
            'cash_flows': cash_flows,
            'assumptions': {
                'discount_rate': self.discount_rate * 100,
                'tax_rate': self.tax_rate * 100,
                'revenue_growth_rate': revenue_growth * 100,
                'cost_inflation_rate': cost_inflation * 100
            }
        }
    
    def _calculate_payback_period(self, cash_flows: List[float]) -> float:
        """Calculate payback period in years"""
        cumulative_cash_flow = 0
        
        for i, cash_flow in enumerate(cash_flows):
            cumulative_cash_flow += cash_flow
            if cumulative_cash_flow >= 0 and i > 0:
                # Interpolate to get exact payback period
                previous_cumulative = cumulative_cash_flow - cash_flow
                return i - 1 + abs(previous_cumulative) / cash_flow
        
        return float('inf')  # Never pays back
    
    def compare_investment_scenarios(self, scenarios: List[Dict[str, any]]) -> pd.DataFrame:
        """Compare multiple investment scenarios"""
        comparison_data = []
        
        for i, scenario in enumerate(scenarios):
            analysis = self.analyze_hospital_investment(**scenario)
            
            comparison_data.append({
                'scenario': scenario.get('name', f'Scenario {i+1}'),
                'initial_investment': scenario['initial_investment'],
                'npv': analysis['investment_summary']['npv'],
                'irr': analysis['investment_summary']['irr'],
                'payback_period': analysis['investment_summary']['payback_period_years'],
                'total_roi': analysis['investment_summary']['total_roi_percentage'],
                'is_profitable': analysis['investment_summary']['is_profitable']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank scenarios by NPV
        comparison_df['npv_rank'] = comparison_df['npv'].rank(ascending=False)
        comparison_df['irr_rank'] = comparison_df['irr'].rank(ascending=False)
        
        return comparison_df.sort_values('npv', ascending=False)

class HealthcareMarketValuation:
    """Market valuation analysis for healthcare companies and assets"""
    
    def __init__(self):
        self.industry_multiples = {
            'hospitals': {'ev_ebitda': 12.5, 'p_e': 18.0, 'p_b': 2.1},
            'pharmaceuticals': {'ev_ebitda': 15.2, 'p_e': 22.5, 'p_b': 3.8},
            'medical_devices': {'ev_ebitda': 18.7, 'p_e': 25.3, 'p_b': 4.2},
            'health_insurance': {'ev_ebitda': 10.8, 'p_e': 14.2, 'p_b': 1.8},
            'healthcare_services': {'ev_ebitda': 14.3, 'p_e': 19.8, 'p_b': 2.9}
        }
    
    def dcf_valuation(self, 
                     free_cash_flows: List[float],
                     terminal_growth_rate: float = 0.025,
                     discount_rate: float = 0.09) -> Dict[str, float]:
        """Discounted Cash Flow valuation"""
        
        logger.info("Performing DCF valuation analysis...")
        
        # Present value of explicit forecast period
        pv_forecast = 0
        for i, fcf in enumerate(free_cash_flows):
            pv_forecast += fcf / ((1 + discount_rate) ** (i + 1))
        
        # Terminal value
        terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))
        
        enterprise_value = pv_forecast + pv_terminal
        
        return {
            'enterprise_value': enterprise_value,
            'pv_forecast_period': pv_forecast,
            'pv_terminal_value': pv_terminal,
            'terminal_value': terminal_value,
            'terminal_growth_rate': terminal_growth_rate * 100,
            'discount_rate': discount_rate * 100
        }
    
    def comparable_company_analysis(self, 
                                  company_metrics: Dict[str, float],
                                  industry_type: str = 'hospitals') -> Dict[str, float]:
        """Comparable company analysis using industry multiples"""
        
        if industry_type not in self.industry_multiples:
            logger.warning(f"Industry type '{industry_type}' not found. Using hospitals as default.")
            industry_type = 'hospitals'
        
        multiples = self.industry_multiples[industry_type]
        valuations = {}
        
        # EV/EBITDA valuation
        if 'ebitda' in company_metrics:
            valuations['ev_ebitda_valuation'] = company_metrics['ebitda'] * multiples['ev_ebitda']
        
        # P/E valuation
        if 'net_income' in company_metrics:
            valuations['pe_valuation'] = company_metrics['net_income'] * multiples['p_e']
        
        # P/B valuation
        if 'book_value' in company_metrics:
            valuations['pb_valuation'] = company_metrics['book_value'] * multiples['p_b']
        
        # Average valuation
        if valuations:
            valuations['average_valuation'] = np.mean(list(valuations.values()))
        
        valuations['industry_multiples_used'] = multiples
        
        return valuations
    
    def healthcare_asset_valuation(self, asset_data: Dict[str, any]) -> Dict[str, any]:
        """Comprehensive healthcare asset valuation"""
        
        asset_type = asset_data.get('type', 'hospital')
        
        valuation_results = {
            'asset_info': {
                'name': asset_data.get('name', 'Healthcare Asset'),
                'type': asset_type,
                'location': asset_data.get('location', 'Unknown'),
                'beds': asset_data.get('beds', 0),
                'annual_revenue': asset_data.get('annual_revenue', 0)
            },
            'valuation_methods': {}
        }
        
        # Asset-based valuation
        if 'total_assets' in asset_data:
            book_value = asset_data['total_assets'] - asset_data.get('total_liabilities', 0)
            market_adjustment = 1.2  # Assume market value 20% above book
            
            valuation_results['valuation_methods']['asset_based'] = {
                'book_value': book_value,
                'adjusted_market_value': book_value * market_adjustment,
                'adjustment_factor': market_adjustment
            }
        
        # Income-based valuation
        if 'annual_revenue' in asset_data and 'operating_margin' in asset_data:
            annual_income = asset_data['annual_revenue'] * asset_data['operating_margin']
            capitalization_rate = 0.12  # 12% cap rate for healthcare assets
            
            valuation_results['valuation_methods']['income_based'] = {
                'annual_income': annual_income,
                'capitalization_rate': capitalization_rate * 100,
                'income_value': annual_income / capitalization_rate
            }
        
        # Market-based valuation (per bed for hospitals)
        if asset_type == 'hospital' and 'beds' in asset_data:
            value_per_bed = 150000  # Industry average $150k per bed
            
            valuation_results['valuation_methods']['market_based'] = {
                'beds': asset_data['beds'],
                'value_per_bed': value_per_bed,
                'total_market_value': asset_data['beds'] * value_per_bed
            }
        
        # Calculate weighted average valuation
        valuations = []
        for method, data in valuation_results['valuation_methods'].items():
            if 'total_market_value' in data:
                valuations.append(data['total_market_value'])
            elif 'income_value' in data:
                valuations.append(data['income_value'])
            elif 'adjusted_market_value' in data:
                valuations.append(data['adjusted_market_value'])
        
        if valuations:
            valuation_results['summary'] = {
                'estimated_value_range': {
                    'low': min(valuations),
                    'high': max(valuations),
                    'average': np.mean(valuations)
                },
                'valuation_methods_used': len(valuations),
                'confidence_level': 'High' if len(valuations) >= 2 else 'Medium'
            }
        
        return valuation_results

class HealthcareInvestmentAnalyzer:
    """Analyzes healthcare investment opportunities and portfolio performance"""
    
    def __init__(self):
        self.risk_free_rate = 0.025  # 2.5% risk-free rate
        
    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = None) -> float:
        """Calculate Sharpe ratio for investment performance"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        excess_returns = [r - risk_free_rate for r in returns]
        
        if len(excess_returns) == 0:
            return 0
        
        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)
        
        return mean_excess_return / std_excess_return if std_excess_return != 0 else 0
    
    def portfolio_optimization(self, 
                             assets: List[Dict[str, any]],
                             target_return: float = None) -> Dict[str, any]:
        """Simple portfolio optimization for healthcare investments"""
        
        logger.info("Performing healthcare portfolio optimization...")
        
        # Extract returns and risks
        expected_returns = [asset['expected_return'] for asset in assets]
        risks = [asset['risk'] for asset in assets]
        asset_names = [asset['name'] for asset in assets]
        
        n_assets = len(assets)
        
        if target_return is None:
            target_return = np.mean(expected_returns)
        
        # Simple equal-weight portfolio as baseline
        equal_weights = [1/n_assets] * n_assets
        
        # Calculate portfolio metrics
        portfolio_return = sum(w * r for w, r in zip(equal_weights, expected_returns))
        portfolio_risk = np.sqrt(sum((w * risk)**2 for w, risk in zip(equal_weights, risks)))
        
        # Risk-adjusted weights (inverse volatility)
        inv_vol_weights = [1/risk for risk in risks]
        total_inv_vol = sum(inv_vol_weights)
        risk_adjusted_weights = [w/total_inv_vol for w in inv_vol_weights]
        
        ra_portfolio_return = sum(w * r for w, r in zip(risk_adjusted_weights, expected_returns))
        ra_portfolio_risk = np.sqrt(sum((w * risk)**2 for w, risk in zip(risk_adjusted_weights, risks)))
        
        return {
            'equal_weight_portfolio': {
                'weights': dict(zip(asset_names, equal_weights)),
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_risk
            },
            'risk_adjusted_portfolio': {
                'weights': dict(zip(asset_names, risk_adjusted_weights)),
                'expected_return': ra_portfolio_return,
                'risk': ra_portfolio_risk,
                'sharpe_ratio': (ra_portfolio_return - self.risk_free_rate) / ra_portfolio_risk
            },
            'individual_assets': [
                {
                    'name': name,
                    'expected_return': ret,
                    'risk': risk,
                    'sharpe_ratio': (ret - self.risk_free_rate) / risk
                }
                for name, ret, risk in zip(asset_names, expected_returns, risks)
            ]
        }
    
    def healthcare_sector_analysis(self, sector_data: Dict[str, List[float]]) -> Dict[str, any]:
        """Analyze different healthcare sectors performance"""
        
        sector_analysis = {}
        
        for sector, returns in sector_data.items():
            if len(returns) > 0:
                sector_analysis[sector] = {
                    'average_return': np.mean(returns),
                    'volatility': np.std(returns),
                    'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                    'max_return': max(returns),
                    'min_return': min(returns),
                    'total_periods': len(returns)
                }
        
        # Rank sectors by Sharpe ratio
        sorted_sectors = sorted(
            sector_analysis.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        return {
            'sector_metrics': sector_analysis,
            'sector_rankings': [
                {
                    'rank': i + 1,
                    'sector': sector,
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'average_return': metrics['average_return'],
                    'volatility': metrics['volatility']
                }
                for i, (sector, metrics) in enumerate(sorted_sectors)
            ]
        }
    
    def investment_risk_assessment(self, investment_data: Dict[str, any]) -> Dict[str, any]:
        """Assess investment risks for healthcare projects"""
        
        risk_factors = {
            'regulatory_risk': investment_data.get('regulatory_complexity', 3),  # 1-5 scale
            'market_risk': investment_data.get('market_volatility', 3),
            'operational_risk': investment_data.get('operational_complexity', 3),
            'financial_risk': investment_data.get('leverage_ratio', 0.3) * 10,  # Convert to 1-5 scale
            'technology_risk': investment_data.get('technology_dependence', 3)
        }
        
        # Calculate overall risk score
        risk_weights = {
            'regulatory_risk': 0.25,
            'market_risk': 0.20,
            'operational_risk': 0.20,
            'financial_risk': 0.20,
            'technology_risk': 0.15
        }
        
        overall_risk_score = sum(
            risk_factors[factor] * risk_weights[factor]
            for factor in risk_factors
        )
        
        # Risk rating
        if overall_risk_score <= 2:
            risk_rating = 'Low'
        elif overall_risk_score <= 3.5:
            risk_rating = 'Medium'
        else:
            risk_rating = 'High'
        
        return {
            'risk_factors': risk_factors,
            'risk_weights': risk_weights,
            'overall_risk_score': overall_risk_score,
            'risk_rating': risk_rating,
            'recommendations': self._generate_risk_recommendations(risk_factors, risk_rating)
        }
    
    def _generate_risk_recommendations(self, risk_factors: Dict[str, float], 
                                     risk_rating: str) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        # High-risk factors (score > 4)
        high_risk_factors = [factor for factor, score in risk_factors.items() if score > 4]
        
        if 'regulatory_risk' in high_risk_factors:
            recommendations.append("Engage regulatory consultants and ensure compliance framework")
        
        if 'market_risk' in high_risk_factors:
            recommendations.append("Implement hedging strategies and diversify market exposure")
        
        if 'operational_risk' in high_risk_factors:
            recommendations.append("Develop robust operational procedures and contingency plans")
        
        if 'financial_risk' in high_risk_factors:
            recommendations.append("Reduce leverage and improve financial stability")
        
        if 'technology_risk' in high_risk_factors:
            recommendations.append("Invest in technology infrastructure and backup systems")
        
        # General recommendations based on overall rating
        if risk_rating == 'High':
            recommendations.append("Consider risk insurance and establish larger contingency reserves")
        elif risk_rating == 'Medium':
            recommendations.append("Monitor risk factors closely and maintain adequate reserves")
        
        return recommendations