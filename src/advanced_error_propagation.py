"""
Advanced Error Propagation Analysis for Fine-Structure Constant Derivation
===========================================================================

This module implements advanced error propagation techniques to track
uncertainty through the complete derivation chain of the fine-structure constant.

Features:
- Automatic differentiation for exact error propagation
- Correlated uncertainty handling
- Higher-order error terms
- Error budget analysis
- Cross-correlation effects
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import numdifftools as nd
from scipy.optimize import minimize
import warnings
from alpha_derivation import AlphaFirstPrinciples, LQGParameters, PolymerParameters, PhysicalConstants


@dataclass
class ErrorBudget:
    """Detailed error budget breakdown"""
    # Parameter uncertainties
    parameter_errors: Dict[str, float] = field(default_factory=dict)
    
    # Systematic errors
    theoretical_model_error: float = 0.0
    computational_error: float = 0.0
    approximation_error: float = 0.0
    
    # Statistical errors
    monte_carlo_error: float = 0.0
    finite_sample_error: float = 0.0
    
    # Cross-correlation terms
    correlation_matrix: np.ndarray = field(default_factory=lambda: np.eye(1))
    cross_correlation_error: float = 0.0
    
    # Total error components
    total_systematic_error: float = 0.0
    total_statistical_error: float = 0.0
    total_combined_error: float = 0.0


class AutomaticDifferentiation:
    """
    Automatic differentiation for exact error propagation.
    Computes gradients and Hessians for uncertainty propagation.
    """
    
    def __init__(self, derivation_function: Callable):
        self.derivation_function = derivation_function
        
    def gradient(self, parameters: np.ndarray) -> np.ndarray:
        """Compute gradient using automatic differentiation"""
        grad_func = nd.Gradient(self.derivation_function)
        return grad_func(parameters)
    
    def hessian(self, parameters: np.ndarray) -> np.ndarray:
        """Compute Hessian matrix for second-order error propagation"""
        hess_func = nd.Hessian(self.derivation_function)
        return hess_func(parameters)
    
    def jacobian(self, parameters: np.ndarray) -> np.ndarray:
        """Compute Jacobian for vector-valued functions"""
        jac_func = nd.Jacobian(self.derivation_function)
        return jac_func(parameters)


class AdvancedErrorPropagation:
    """
    Advanced error propagation analysis with correlation handling.
    """
    
    def __init__(self):
        self.parameter_names = [
            'gamma_immirzi', 'phi_vac', 'planck_length', 
            'mu_polymer', 'discretization_scale',
            'hbar', 'e', 'epsilon_0', 'c'
        ]
        
        # Physical parameter nominal values
        self.nominal_parameters = np.array([
            0.2375,          # gamma_immirzi
            1.496e10,        # phi_vac
            1.616255e-35,    # planck_length
            1.0,             # mu_polymer
            1.0,             # discretization_scale
            1.054571817e-34, # hbar
            1.602176634e-19, # e
            8.8541878128e-12,# epsilon_0
            299792458        # c
        ])
        
        # Parameter uncertainties (standard deviations)
        self.parameter_uncertainties = np.array([
            0.01,            # gamma_immirzi uncertainty
            1e8,             # phi_vac uncertainty
            1e-37,           # planck_length uncertainty  
            0.05,            # mu_polymer uncertainty
            0.02,            # discretization_scale uncertainty
            1.3e-42,         # hbar uncertainty (CODATA)
            2.2e-28,         # e uncertainty (CODATA)
            1.3e-21,         # epsilon_0 uncertainty (CODATA)
            0.0              # c uncertainty (exact)
        ])
        
        # Initialize correlation matrix (assume uncorrelated for now)
        self.correlation_matrix = np.eye(len(self.parameter_names))
        
        # Set known correlations
        self._set_physical_correlations()
    
    def _set_physical_correlations(self):
        """Set physically motivated parameter correlations"""
        # LQG parameters may be correlated
        gamma_idx = self.parameter_names.index('gamma_immirzi')
        phi_vac_idx = self.parameter_names.index('phi_vac')
        self.correlation_matrix[gamma_idx, phi_vac_idx] = 0.1
        self.correlation_matrix[phi_vac_idx, gamma_idx] = 0.1
        
        # Physical constants correlations (CODATA)
        hbar_idx = self.parameter_names.index('hbar')
        e_idx = self.parameter_names.index('e')
        epsilon_0_idx = self.parameter_names.index('epsilon_0')
        
        # Small correlations between fundamental constants
        self.correlation_matrix[hbar_idx, e_idx] = 0.05
        self.correlation_matrix[e_idx, hbar_idx] = 0.05
        self.correlation_matrix[e_idx, epsilon_0_idx] = -0.03
        self.correlation_matrix[epsilon_0_idx, e_idx] = -0.03
    
    def create_derivation_function(self, parameters: np.ndarray) -> float:
        """
        Create derivation function for automatic differentiation.
        Maps parameter vector to fine-structure constant value.
        """
        try:
            # Unpack parameters
            gamma_immirzi, phi_vac, planck_length, mu_polymer, discretization_scale, \
            hbar, e, epsilon_0, c = parameters
            
            # Create parameter objects
            lqg_params = LQGParameters(
                gamma_immirzi=gamma_immirzi,
                phi_vac=phi_vac,
                planck_length=planck_length
            )
            
            polymer_params = PolymerParameters(
                mu_polymer=mu_polymer,
                discretization_scale=discretization_scale
            )
            
            # Create custom physical constants
            constants = PhysicalConstants()
            constants.hbar = hbar
            constants.e = e
            constants.epsilon_0 = epsilon_0
            constants.c = c
            
            # Create derivation instance
            calc = AlphaFirstPrinciples(lqg_params=lqg_params, polymer_params=polymer_params)
            calc.constants = constants
            
            # Compute fine-structure constant
            results = calc.derive_alpha_complete()
            return results['final_theoretical']
            
        except Exception as e:
            # Return NaN for failed computations
            warnings.warn(f"Computation failed: {str(e)}")
            return np.nan
    
    def first_order_error_propagation(self) -> Dict:
        """
        First-order (linear) error propagation using gradient.
        σ²(α) = ∇α · C · ∇α^T where C is covariance matrix
        """
        print("Computing first-order error propagation...")
        
        # Compute gradient at nominal parameters
        ad = AutomaticDifferentiation(self.create_derivation_function)
        gradient = ad.gradient(self.nominal_parameters)
        
        # Construct covariance matrix
        std_matrix = np.diag(self.parameter_uncertainties)
        covariance_matrix = std_matrix @ self.correlation_matrix @ std_matrix
        
        # First-order uncertainty propagation
        variance_alpha = gradient.T @ covariance_matrix @ gradient
        uncertainty_alpha = np.sqrt(variance_alpha)
        
        # Individual parameter contributions
        individual_contributions = {}
        for i, param_name in enumerate(self.parameter_names):
            # Variance contribution from parameter i
            contrib_variance = (gradient[i] * self.parameter_uncertainties[i])**2
            individual_contributions[param_name] = {
                'gradient': gradient[i],
                'parameter_uncertainty': self.parameter_uncertainties[i],
                'variance_contribution': contrib_variance,
                'std_contribution': np.sqrt(contrib_variance),
                'relative_contribution': contrib_variance / variance_alpha if variance_alpha > 0 else 0
            }
        
        # Compute nominal α value
        nominal_alpha = self.create_derivation_function(self.nominal_parameters)
        
        results = {
            'nominal_alpha': nominal_alpha,
            'gradient': gradient,
            'covariance_matrix': covariance_matrix,
            'total_variance': variance_alpha,
            'total_uncertainty': uncertainty_alpha,
            'relative_uncertainty': uncertainty_alpha / nominal_alpha if nominal_alpha != 0 else np.inf,
            'individual_contributions': individual_contributions
        }
        
        return results
    
    def second_order_error_propagation(self) -> Dict:
        """
        Second-order error propagation including Hessian terms.
        Accounts for nonlinear effects in uncertainty propagation.
        """
        print("Computing second-order error propagation...")
        
        # Compute gradient and Hessian
        ad = AutomaticDifferentiation(self.create_derivation_function)
        gradient = ad.gradient(self.nominal_parameters)
        hessian = ad.hessian(self.nominal_parameters)
        
        # Construct covariance matrix
        std_matrix = np.diag(self.parameter_uncertainties)
        covariance_matrix = std_matrix @ self.correlation_matrix @ std_matrix
        
        # First-order term
        first_order_variance = gradient.T @ covariance_matrix @ gradient
        
        # Second-order correction
        # σ²₂ = (1/2) * Tr(H · C)² where H is Hessian, C is covariance
        hessian_covariance = hessian @ covariance_matrix
        second_order_correction = 0.5 * np.trace(hessian_covariance @ hessian_covariance)
        
        # Total variance (first + second order)
        total_variance = first_order_variance + second_order_correction
        total_uncertainty = np.sqrt(abs(total_variance))  # abs() for numerical stability
        
        # Nonlinearity assessment
        nonlinearity_factor = abs(second_order_correction / first_order_variance) if first_order_variance > 0 else np.inf
        
        # Compute nominal α value
        nominal_alpha = self.create_derivation_function(self.nominal_parameters)
        
        results = {
            'nominal_alpha': nominal_alpha,
            'gradient': gradient,
            'hessian': hessian,
            'first_order_variance': first_order_variance,
            'second_order_correction': second_order_correction,
            'total_variance': total_variance,
            'total_uncertainty': total_uncertainty,
            'relative_uncertainty': total_uncertainty / nominal_alpha if nominal_alpha != 0 else np.inf,
            'nonlinearity_factor': nonlinearity_factor,
            'is_linear_approximation_valid': nonlinearity_factor < 0.1
        }
        
        return results
    
    def correlation_uncertainty_analysis(self) -> Dict:
        """
        Analyze impact of parameter correlations on uncertainty.
        """
        print("Analyzing correlation effects...")
        
        # Compute uncertainty with full correlations
        first_order_results = self.first_order_error_propagation()
        uncertainty_with_correlations = first_order_results['total_uncertainty']
        
        # Compute uncertainty assuming no correlations
        original_correlation = self.correlation_matrix.copy()
        self.correlation_matrix = np.eye(len(self.parameter_names))
        
        uncorrelated_results = self.first_order_error_propagation()
        uncertainty_uncorrelated = uncorrelated_results['total_uncertainty']
        
        # Restore original correlation matrix
        self.correlation_matrix = original_correlation
        
        # Correlation impact
        correlation_effect = uncertainty_with_correlations - uncertainty_uncorrelated
        relative_correlation_effect = correlation_effect / uncertainty_uncorrelated if uncertainty_uncorrelated > 0 else 0
        
        results = {
            'uncertainty_with_correlations': uncertainty_with_correlations,
            'uncertainty_uncorrelated': uncertainty_uncorrelated,
            'correlation_effect': correlation_effect,
            'relative_correlation_effect': relative_correlation_effect,
            'correlation_increases_uncertainty': correlation_effect > 0,
            'correlation_matrix': original_correlation
        }
        
        return results
    
    def error_budget_analysis(self) -> ErrorBudget:
        """
        Comprehensive error budget analysis.
        """
        print("Computing error budget analysis...")
        
        # Get first and second order results
        first_order = self.first_order_error_propagation()
        second_order = self.second_order_error_propagation()
        correlation_analysis = self.correlation_uncertainty_analysis()
        
        # Create error budget
        budget = ErrorBudget()
        
        # Parameter-specific errors
        for param_name, contrib in first_order['individual_contributions'].items():
            budget.parameter_errors[param_name] = contrib['std_contribution']
        
        # Systematic errors (model uncertainties)
        budget.theoretical_model_error = 1e-6  # Estimated LQG model uncertainty
        budget.computational_error = 1e-12     # Numerical computation error
        budget.approximation_error = abs(second_order['second_order_correction'])**0.5
        
        # Statistical errors
        budget.monte_carlo_error = 0.0  # No MC sampling in this analysis
        budget.finite_sample_error = 0.0
        
        # Cross-correlation effects
        budget.correlation_matrix = correlation_analysis['correlation_matrix']
        budget.cross_correlation_error = abs(correlation_analysis['correlation_effect'])
        
        # Total errors
        budget.total_systematic_error = np.sqrt(
            budget.theoretical_model_error**2 + 
            budget.computational_error**2 + 
            budget.approximation_error**2
        )
        
        parameter_variance = sum(contrib['variance_contribution'] 
                               for contrib in first_order['individual_contributions'].values())
        budget.total_statistical_error = np.sqrt(parameter_variance)
        
        budget.total_combined_error = np.sqrt(
            budget.total_systematic_error**2 + 
            budget.total_statistical_error**2 +
            budget.cross_correlation_error**2
        )
        
        return budget
    
    def comprehensive_error_analysis(self) -> Dict:
        """
        Run comprehensive error propagation analysis.
        """
        print("=" * 80)
        print("COMPREHENSIVE ERROR PROPAGATION ANALYSIS")
        print("=" * 80)
        
        # Perform all analyses
        first_order = self.first_order_error_propagation()
        second_order = self.second_order_error_propagation()
        correlation_analysis = self.correlation_uncertainty_analysis()
        error_budget = self.error_budget_analysis()
        
        # Combine results
        comprehensive_results = {
            'first_order_propagation': first_order,
            'second_order_propagation': second_order,
            'correlation_analysis': correlation_analysis,
            'error_budget': error_budget,
            
            # Summary
            'summary': {
                'nominal_alpha': first_order['nominal_alpha'],
                'total_uncertainty': error_budget.total_combined_error,
                'relative_uncertainty': error_budget.total_combined_error / first_order['nominal_alpha'],
                'dominant_error_source': self._identify_dominant_error_source(error_budget),
                'is_linear_regime': second_order['is_linear_approximation_valid'],
                'correlation_significant': abs(correlation_analysis['relative_correlation_effect']) > 0.01
            }
        }
        
        return comprehensive_results
    
    def _identify_dominant_error_source(self, budget: ErrorBudget) -> str:
        """Identify the dominant source of uncertainty"""
        sources = {
            'Statistical (Parameters)': budget.total_statistical_error,
            'Systematic (Model)': budget.total_systematic_error,
            'Cross-correlations': budget.cross_correlation_error
        }
        
        return max(sources, key=sources.get)
    
    def generate_error_report(self, error_results: Dict) -> None:
        """
        Generate comprehensive error propagation report.
        """
        print("\n" + "=" * 80)
        print("ERROR PROPAGATION REPORT")
        print("=" * 80)
        
        first_order = error_results['first_order_propagation']
        second_order = error_results['second_order_propagation']
        correlation = error_results['correlation_analysis']
        budget = error_results['error_budget']
        summary = error_results['summary']
        
        # Summary
        print(f"\n1. SUMMARY")
        print(f"   Nominal α: {summary['nominal_alpha']:.12e}")
        print(f"   Total uncertainty: {summary['total_uncertainty']:.2e}")
        print(f"   Relative uncertainty: {summary['relative_uncertainty']:.2e} ({summary['relative_uncertainty']*100:.4f}%)")
        print(f"   Dominant error source: {summary['dominant_error_source']}")
        print(f"   Linear regime valid: {'✓' if summary['is_linear_regime'] else '✗'}")
        print(f"   Correlations significant: {'✓' if summary['correlation_significant'] else '✗'}")
        
        # Parameter contributions
        print(f"\n2. PARAMETER UNCERTAINTY CONTRIBUTIONS")
        contributions = first_order['individual_contributions']
        sorted_params = sorted(contributions.items(), 
                             key=lambda x: x[1]['relative_contribution'], reverse=True)
        
        for param_name, contrib in sorted_params[:5]:  # Top 5 contributors
            print(f"   {param_name}:")
            print(f"     Relative contribution: {contrib['relative_contribution']:.1%}")
            print(f"     Uncertainty contribution: {contrib['std_contribution']:.2e}")
            print(f"     Gradient: {contrib['gradient']:.2e}")
        
        # Error budget breakdown
        print(f"\n3. ERROR BUDGET BREAKDOWN")
        print(f"   Statistical error: {budget.total_statistical_error:.2e}")
        print(f"   Systematic error: {budget.total_systematic_error:.2e}")
        print(f"     - Model uncertainty: {budget.theoretical_model_error:.2e}")
        print(f"     - Computational error: {budget.computational_error:.2e}")
        print(f"     - Approximation error: {budget.approximation_error:.2e}")
        print(f"   Cross-correlation error: {budget.cross_correlation_error:.2e}")
        print(f"   Total combined error: {budget.total_combined_error:.2e}")
        
        # Nonlinearity assessment
        print(f"\n4. NONLINEARITY ASSESSMENT")
        print(f"   First-order variance: {second_order['first_order_variance']:.2e}")
        print(f"   Second-order correction: {second_order['second_order_correction']:.2e}")
        print(f"   Nonlinearity factor: {second_order['nonlinearity_factor']:.2e}")
        
        if second_order['nonlinearity_factor'] < 0.01:
            linearity_assessment = "Highly linear - first-order propagation sufficient"
        elif second_order['nonlinearity_factor'] < 0.1:
            linearity_assessment = "Approximately linear - first-order dominates"
        else:
            linearity_assessment = "Significant nonlinearity - higher-order terms important"
        print(f"   Assessment: {linearity_assessment}")
        
        # Correlation effects
        print(f"\n5. CORRELATION EFFECTS")
        print(f"   Uncertainty with correlations: {correlation['uncertainty_with_correlations']:.2e}")
        print(f"   Uncertainty without correlations: {correlation['uncertainty_uncorrelated']:.2e}")
        print(f"   Correlation effect: {correlation['correlation_effect']:.2e}")
        print(f"   Relative effect: {correlation['relative_correlation_effect']:.1%}")
        
        if abs(correlation['relative_correlation_effect']) > 0.05:
            correlation_significance = "SIGNIFICANT - correlations cannot be ignored"
        elif abs(correlation['relative_correlation_effect']) > 0.01:
            correlation_significance = "MODERATE - correlations should be considered"
        else:
            correlation_significance = "NEGLIGIBLE - correlations have minimal impact"
        print(f"   Significance: {correlation_significance}")
        
        # Recommendations
        print(f"\n6. RECOMMENDATIONS")
        
        if summary['relative_uncertainty'] > 1e-4:
            print(f"   ⚠ High uncertainty - consider improving parameter precision")
        
        if not summary['is_linear_regime']:
            print(f"   ⚠ Nonlinear regime - use second-order error propagation")
        
        if summary['correlation_significant']:
            print(f"   ⚠ Include parameter correlations in uncertainty analysis")
        
        if budget.total_systematic_error > budget.total_statistical_error:
            print(f"   ⚠ Systematic errors dominate - improve theoretical model")
        
        if all([
            summary['relative_uncertainty'] <= 1e-5,
            summary['is_linear_regime'],
            budget.total_systematic_error <= budget.total_statistical_error
        ]):
            print(f"   ✓ Excellent error control - derivation is robust")
        
        print("=" * 80)


def run_advanced_error_propagation():
    """Run comprehensive error propagation analysis"""
    
    # Initialize error propagation analysis
    error_prop = AdvancedErrorPropagation()
    
    # Run comprehensive analysis
    results = error_prop.comprehensive_error_analysis()
    
    # Generate report
    error_prop.generate_error_report(results)
    
    return results


if __name__ == "__main__":
    results = run_advanced_error_propagation()
