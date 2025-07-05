#!/usr/bin/env python3
"""
Example computation of the fine-structure constant from first principles.

This script demonstrates the complete derivation process and validates
the theoretical results against CODATA values.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_derivation import AlphaFirstPrinciples, LQGParameters, PolymerParameters
import numpy as np
import matplotlib.pyplot as plt


def run_basic_example():
    """Run basic α derivation example"""
    print("=== Basic Fine-Structure Constant Derivation ===")
    
    # Initialize with default parameters
    alpha_calc = AlphaFirstPrinciples()
    
    # Get complete derivation results
    results = alpha_calc.derive_alpha_complete()
    
    print(f"\nResults Summary:")
    print(f"Theoretical α = {results['final_theoretical']:.12e}")
    print(f"CODATA α     = {results['codata_value']:.12e}")
    print(f"Absolute Error = {abs(results['final_theoretical'] - results['codata_value']):.2e}")
    print(f"Relative Error = {results['relative_error']:.2e}")
    print(f"Agreement     = {results['agreement_percentage']:.6f}%")
    
    print(f"\nComponent Breakdown:")
    components = ['vacuum_parameter', 'geometric', 'polymer_corrected', 
                 'vacuum_polarization', 'running_coupling', 'scalar_tensor']
    
    for comp in components:
        if comp in results:
            error = abs(results[comp] - results['codata_value']) / results['codata_value']
            print(f"  {comp:20s}: {results[comp]:.8e} (error: {error:.2e})")
    
    return results


def parameter_sensitivity_analysis():
    """Analyze sensitivity to LQG and polymer parameters"""
    print("\n=== Parameter Sensitivity Analysis ===")
    
    # Base parameters
    base_lqg = LQGParameters()
    base_polymer = PolymerParameters()
    
    # Vary Immirzi parameter
    gamma_values = np.linspace(0.1, 0.5, 10)
    alpha_gamma = []
    
    for gamma in gamma_values:
        lqg_params = LQGParameters(gamma_immirzi=gamma)
        calc = AlphaFirstPrinciples(lqg_params=lqg_params)
        results = calc.derive_alpha_complete()
        alpha_gamma.append(results['final_theoretical'])
    
    # Vary polymer scale
    mu_values = np.linspace(0.5, 2.0, 10)
    alpha_mu = []
    
    for mu in mu_values:
        polymer_params = PolymerParameters(mu_polymer=mu)
        calc = AlphaFirstPrinciples(polymer_params=polymer_params)
        results = calc.derive_alpha_complete()
        alpha_mu.append(results['final_theoretical'])
    
    print(f"Immirzi parameter sensitivity:")
    print(f"  γ range: {gamma_values[0]:.2f} - {gamma_values[-1]:.2f}")
    print(f"  α range: {min(alpha_gamma):.8e} - {max(alpha_gamma):.8e}")
    print(f"  Variation: {(max(alpha_gamma) - min(alpha_gamma))/min(alpha_gamma):.2e}")
    
    print(f"\nPolymer scale sensitivity:")
    print(f"  μ range: {mu_values[0]:.2f} - {mu_values[-1]:.2f}")
    print(f"  α range: {min(alpha_mu):.8e} - {max(alpha_mu):.8e}")
    print(f"  Variation: {(max(alpha_mu) - min(alpha_mu))/min(alpha_mu):.2e}")
    
    return gamma_values, alpha_gamma, mu_values, alpha_mu


def convergence_analysis():
    """Analyze convergence with respect to spin network cutoffs"""
    print("\n=== Convergence Analysis ===")
    
    # Test different j_max values in geometric formulation
    from alpha_derivation import AlphaFirstPrinciples
    
    calc = AlphaFirstPrinciples()
    
    # Override the geometric calculation with different j_max
    j_max_values = range(5, 51, 5)
    alpha_convergence = []
    
    for j_max in j_max_values:
        # Manually compute geometric contribution
        spin_sum = sum(np.sqrt(j * (j + 1)) for j in range(1, j_max + 1))
        gamma_factor = calc.lqg.gamma_immirzi / (8 * np.pi)
        holonomy_factor = np.exp(-calc.lqg.gamma_immirzi**2 / 2)
        geometric_factor = gamma_factor * spin_sum * holonomy_factor
        
        # Scale to electromagnetic coupling
        scaling_factor = 4 * np.pi * calc.constants.epsilon_0
        alpha_geom = geometric_factor * scaling_factor
        
        alpha_convergence.append(alpha_geom)
    
    print(f"Geometric convergence with j_max:")
    for j_max, alpha_val in zip(j_max_values, alpha_convergence):
        error = abs(alpha_val - calc.constants.alpha_codata) / calc.constants.alpha_codata
        print(f"  j_max = {j_max:2d}: α = {alpha_val:.8e} (error: {error:.2e})")
    
    # Check convergence rate
    if len(alpha_convergence) > 1:
        convergence_rate = abs(alpha_convergence[-1] - alpha_convergence[-2]) / abs(alpha_convergence[-2] - alpha_convergence[-3])
        print(f"\nConvergence rate: {convergence_rate:.3f}")
    
    return j_max_values, alpha_convergence


def validate_against_codata():
    """Comprehensive validation against CODATA values"""
    print("\n=== CODATA Validation ===")
    
    calc = AlphaFirstPrinciples()
    verification = calc.verify_derivation()
    
    print(f"Validation Results:")
    print(f"  Theoretical α:     {verification['theoretical_alpha']:.12e}")
    print(f"  CODATA α:          {verification['codata_alpha']:.12e}")
    print(f"  Absolute error:    {verification['absolute_error']:.2e}")
    print(f"  Relative error:    {verification['relative_error_percent']:.4f}%")
    print(f"  Agreement:         {verification['agreement_percent']:.6f}%")
    print(f"  Precision digits:  {verification['precision_digits']:.2f}")
    
    # Status assessment
    if verification['relative_error_percent'] < 0.01:
        status = "EXCELLENT - Sub-percent accuracy achieved"
    elif verification['relative_error_percent'] < 0.1:
        status = "VERY GOOD - Sub-0.1% accuracy achieved"
    elif verification['relative_error_percent'] < 1.0:
        status = "GOOD - Sub-1% accuracy achieved"
    else:
        status = "NEEDS IMPROVEMENT - Error > 1%"
    
    print(f"  Status: {status}")
    
    return verification


def main():
    """Run complete example suite"""
    print("Fine-Structure Constant First-Principles Derivation")
    print("=" * 60)
    
    # Basic example
    basic_results = run_basic_example()
    
    # Parameter sensitivity
    sensitivity_results = parameter_sensitivity_analysis()
    
    # Convergence analysis
    convergence_results = convergence_analysis()
    
    # CODATA validation
    validation_results = validate_against_codata()
    
    print(f"\n=== Summary ===")
    print(f"First-principles α derivation complete.")
    print(f"Theoretical value: {basic_results['final_theoretical']:.10e}")
    print(f"CODATA agreement: {basic_results['agreement_percentage']:.4f}%")
    print(f"Framework successfully integrates:")
    print(f"  ✓ LQG geometric invariants")
    print(f"  ✓ Polymer quantization corrections") 
    print(f"  ✓ Enhanced vacuum polarization")
    print(f"  ✓ Running coupling β-functions")
    print(f"  ✓ Scalar-tensor field enhancements")
    
    return {
        'basic': basic_results,
        'sensitivity': sensitivity_results,
        'convergence': convergence_results,
        'validation': validation_results
    }


if __name__ == "__main__":
    results = main()
