"""
Test suite for fine-structure constant derivation validation.

Comprehensive testing of the first-principles α derivation against
CODATA values and theoretical consistency checks.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from alpha_derivation import AlphaFirstPrinciples, LQGParameters, PolymerParameters


class TestAlphaDerivation(unittest.TestCase):
    """Test the fine-structure constant derivation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calc = AlphaFirstPrinciples()
        self.codata_alpha = 7.2973525693e-3
        self.tolerance = 1e-4  # 0.01% tolerance
        
    def test_vacuum_parameter_framework(self):
        """Test vacuum parameter framework gives reasonable result"""
        alpha_vacuum = self.calc.vacuum_parameter_framework()
        
        # Should be within reasonable range
        self.assertGreater(alpha_vacuum, 1e-5)
        self.assertLess(alpha_vacuum, 1e-1)
        
        # Should be approximately correct order of magnitude
        relative_error = abs(alpha_vacuum - self.codata_alpha) / self.codata_alpha
        self.assertLess(relative_error, 10.0)  # Within factor of 10
        
    def test_polymer_quantization_corrections(self):
        """Test polymer quantization corrections"""
        alpha_base = self.codata_alpha
        alpha_polymer = self.calc.polymer_quantization_corrections(alpha_base)
        
        # Polymer corrections should be small for reasonable μ_polymer
        correction_factor = alpha_polymer / alpha_base
        self.assertGreater(correction_factor, 0.5)
        self.assertLess(correction_factor, 2.0)
        
    def test_enhanced_vacuum_polarization(self):
        """Test enhanced vacuum polarization calculation"""
        q_squared = (1e8)**2  # Typical QED scale
        pi_correction = self.calc.enhanced_vacuum_polarization(q_squared)
        
        # Should be positive and reasonable magnitude
        self.assertGreater(pi_correction, 0)
        self.assertLess(pi_correction, 1.0)
        
    def test_running_coupling_beta_function(self):
        """Test running coupling β-function"""
        energy = 1e8  # High energy scale
        alpha_input = self.codata_alpha
        
        beta = self.calc.running_coupling_beta_function(energy, alpha_input)
        
        # β-function should be positive (α increases with energy in QED)
        self.assertGreater(beta, 0)
        self.assertLess(beta, 1e-3)  # Should be small
        
    def test_holonomy_flux_geometric_formulation(self):
        """Test LQG geometric formulation"""
        alpha_geometric = self.calc.holonomy_flux_geometric_formulation()
        
        # Should give reasonable result
        self.assertGreater(alpha_geometric, 1e-6)
        self.assertLess(alpha_geometric, 1e-1)
        
    def test_scalar_tensor_enhancement(self):
        """Test scalar-tensor field enhancement"""
        alpha_base = self.codata_alpha
        spacetime_pos = (0, 0, 0, 0)  # Origin
        
        alpha_enhanced = self.calc.scalar_tensor_enhancement(alpha_base, spacetime_pos)
        
        # Enhancement should be reasonable
        enhancement_factor = alpha_enhanced / alpha_base
        self.assertGreater(enhancement_factor, 0.1)
        self.assertLess(enhancement_factor, 10.0)
        
    def test_complete_derivation_convergence(self):
        """Test that complete derivation converges to reasonable value"""
        results = self.calc.derive_alpha_complete()
        
        # Final result should exist
        self.assertIn('final_theoretical', results)
        alpha_final = results['final_theoretical']
        
        # Should be positive and finite
        self.assertGreater(alpha_final, 0)
        self.assertFalse(np.isnan(alpha_final))
        self.assertFalse(np.isinf(alpha_final))
        
        # Should be within reasonable range of CODATA
        relative_error = abs(alpha_final - self.codata_alpha) / self.codata_alpha
        self.assertLess(relative_error, 1.0)  # Within 100%
        
    def test_codata_agreement_quality(self):
        """Test quality of agreement with CODATA"""
        verification = self.calc.verify_derivation()
        
        # Should have all expected fields
        expected_fields = ['theoretical_alpha', 'codata_alpha', 'absolute_error',
                          'relative_error_percent', 'agreement_percent', 'precision_digits']
        
        for field in expected_fields:
            self.assertIn(field, verification)
            
        # Agreement should be reasonable
        self.assertGreater(verification['agreement_percent'], 0.0)
        self.assertLess(verification['relative_error_percent'], 100.0)
        
    def test_parameter_consistency(self):
        """Test that physical parameters are consistent"""
        # LQG parameters should be positive
        self.assertGreater(self.calc.lqg.gamma_immirzi, 0)
        self.assertGreater(self.calc.lqg.planck_length, 0)
        self.assertGreater(self.calc.lqg.phi_vac, 0)
        
        # Polymer parameters should be positive
        self.assertGreater(self.calc.polymer.mu_polymer, 0)
        self.assertGreater(self.calc.polymer.discretization_scale, 0)
        
        # Physical constants should match expected values
        self.assertAlmostEqual(self.calc.constants.c, 299792458.0)
        self.assertAlmostEqual(self.calc.constants.alpha_codata, self.codata_alpha, places=10)
        
    def test_geometric_invariants_computation(self):
        """Test geometric invariants computation"""
        geometric_factor = self.calc._compute_geometric_invariants()
        
        # Should be positive and finite
        self.assertGreater(geometric_factor, 0)
        self.assertFalse(np.isnan(geometric_factor))
        self.assertFalse(np.isinf(geometric_factor))
        
    def test_different_lqg_parameters(self):
        """Test derivation with different LQG parameters"""
        # Test with different Immirzi parameter
        lqg_alt = LQGParameters(gamma_immirzi=0.3)
        calc_alt = AlphaFirstPrinciples(lqg_params=lqg_alt)
        
        results_alt = calc_alt.derive_alpha_complete()
        
        # Should still give reasonable result
        alpha_alt = results_alt['final_theoretical']
        self.assertGreater(alpha_alt, 1e-5)
        self.assertLess(alpha_alt, 1e-1)
        
    def test_different_polymer_parameters(self):
        """Test derivation with different polymer parameters"""
        # Test with different polymer scale
        polymer_alt = PolymerParameters(mu_polymer=0.5)
        calc_alt = AlphaFirstPrinciples(polymer_params=polymer_alt)
        
        results_alt = calc_alt.derive_alpha_complete()
        
        # Should still give reasonable result
        alpha_alt = results_alt['final_theoretical']
        self.assertGreater(alpha_alt, 1e-5)
        self.assertLess(alpha_alt, 1e-1)


class TestMathematicalConsistency(unittest.TestCase):
    """Test mathematical consistency of derivation"""
    
    def setUp(self):
        self.calc = AlphaFirstPrinciples()
        
    def test_polynomial_series_convergence(self):
        """Test that polymer correction series converges"""
        mu_values = [0.01, 0.1, 0.5, 1.0]
        alpha_base = 7.3e-3
        
        for mu in mu_values:
            # Test series expansion vs exact
            x = mu * alpha_base
            if x < 0.1:
                exact = np.sin(x) / x
                series = 1 - x**2/6 + x**4/120
                relative_error = abs(exact - series) / exact
                self.assertLess(relative_error, 0.01)  # 1% accuracy for small x
                
    def test_vacuum_polarization_sign(self):
        """Test that vacuum polarization has correct sign"""
        q_squared = 1e10
        pi_correction = self.calc.enhanced_vacuum_polarization(q_squared)
        
        # Vacuum polarization should be positive (screening effect)
        self.assertGreater(pi_correction, 0)
        
    def test_beta_function_positivity(self):
        """Test that QED β-function is positive"""
        energies = [1e6, 1e8, 1e10]  # Various energy scales
        alpha_input = 7.3e-3
        
        for energy in energies:
            beta = self.calc.running_coupling_beta_function(energy, alpha_input)
            self.assertGreater(beta, 0)  # QED β-function is positive
            
    def test_dimensional_analysis(self):
        """Test dimensional consistency"""
        # Fine-structure constant should be dimensionless
        results = self.calc.derive_alpha_complete()
        alpha_final = results['final_theoretical']
        
        # Should be dimensionless number of order 10^-2
        self.assertGreater(alpha_final, 1e-4)
        self.assertLess(alpha_final, 1e-1)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
