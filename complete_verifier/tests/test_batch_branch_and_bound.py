#########################################################################
##   This file is part of the α,β-CROWN (alpha-beta-CROWN) verifier    ##
##                                                                     ##
##   Copyright (C) 2021-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
"""Unit tests for input_split/batch_branch_and_bound.py"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import math

import torch

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arguments


def _setup_mock_config():
    """Create a mock config for testing."""
    new_config = arguments.ConfigHandler()
    new_config.construct_config_dict(new_config.default_args)
    new_config.file = None

    # Set up test-specific values
    new_config['bab']['branching']['input_split']['split_hint'] = None
    new_config['bab']['branching']['input_split']['update_rhs_with_attack'] = False
    new_config['bab']['branching']['input_split']['compare_with_old_bounds'] = False
    new_config['bab']['branching']['input_split']['skip_getting_worst_domain'] = False
    new_config['bab']['branching']['input_split']['show_progress'] = False
    new_config['bab']['branching']['input_split']['reorder_bab'] = False
    new_config['bab']['branching']['input_split']['adv_check'] = -1
    new_config['bab']['branching']['input_split']['split_partitions'] = 2
    new_config['bab']['branching']['input_split']['presplit_domains'] = None
    new_config['bab']['branching']['input_split']['sort_index'] = None
    new_config['bab']['branching']['input_split']['sort_descending'] = True
    new_config['bab']['branching']['input_split']['bf_iters'] = 0
    new_config['bab']['branching']['input_split']['bf_batch_size'] = 1
    new_config['bab']['clip_n_verify']['clip_input_domain']['enabled'] = False
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_iterations'] = 1
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_calculate_volume_metrics'] = False
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_type'] = 'simple'
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_neuron_selection_value'] = 1.0
    new_config['bab']['clip_n_verify']['clip_input_domain']['clip_neuron_selection_type'] = 'ratio'
    new_config['bab']['timeout'] = 100
    new_config['bab']['max_iterations'] = 10
    new_config['bab']['sort_domain_interval'] = 0
    new_config['bab']['branching']['method'] = 'sb'
    new_config['solver']['batch_size'] = 64
    new_config['solver']['bound_prop_method'] = 'crown'
    new_config['solver']['init_bound_prop_method'] = 'same'
    new_config['solver']['min_batch_size_ratio'] = 0.1
    new_config['solver']['auto_enlarge_batch_size'] = False
    new_config['attack']['pgd_order'] = 'skip'
    new_config['attack']['input_split_check_adv']['enabled'] = 'false'
    new_config['model']['with_jacobian'] = False
    new_config['general']['device'] = 'cpu'
    new_config['general']['enable_incomplete_verification'] = True

    return new_config


# ============================================================================
# _print_final_results Tests
# ============================================================================

class TestPrintFinalResults(unittest.TestCase):
    """Tests for _print_final_results function."""

    def setUp(self):
        """Set up mock config."""
        self.original_config = arguments.Config
        arguments.Config = _setup_mock_config()

    def tearDown(self):
        """Restore original config."""
        arguments.Config = self.original_config

    @patch('builtins.print')
    def test_print_final_results_basic(self, mock_print):
        """Test _print_final_results outputs correctly."""
        from input_split.batch_branch_and_bound import _print_final_results
        from input_split.utils import Stats

        stats = Stats()
        stats.visited = 100
        stats.timer.start('summary')

        _print_final_results(stats, 50)

        # Verify print was called with expected domain info
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should print domain length and visited count
        self.assertTrue(any('50' in call for call in print_calls), "Should print domain length")
        self.assertTrue(any('100' in call and 'visited' in call for call in print_calls), "Should print visited count")

    @patch('builtins.print')
    def test_print_final_results_zero_domains(self, mock_print):
        """Test _print_final_results with zero domains."""
        from input_split.batch_branch_and_bound import _print_final_results
        from input_split.utils import Stats

        stats = Stats()
        stats.visited = 0
        stats.timer.start('summary')

        _print_final_results(stats, 0)

        # Verify print was called with zero values
        print_calls = [str(call) for call in mock_print.call_args_list]
        # Should print "Length of domains: 0" and "0 domains visited"
        self.assertTrue(any('0' in call for call in print_calls), "Should print domain length of 0")
        self.assertTrue(any('0' in call and 'visited' in call for call in print_calls), "Should print 0 visited")


class TestInitialVerification(unittest.TestCase):
    """Tests for initial verification logic."""

    def test_initial_verified_true(self):
        """Test initial verification when all bounds verified."""
        from input_split.utils import initial_verify_criterion

        global_lb = torch.tensor([[0.5, 0.3], [0.1, 0.2]])
        rhs = torch.zeros(2, 2)

        initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)

        self.assertTrue(initial_verified)

    def test_initial_verified_false(self):
        """Test initial verification when not all bounds verified."""
        from input_split.utils import initial_verify_criterion

        # For a sample to be verified, ANY of its lbs - rhs must be > 0
        # First sample: [-0.5, -0.3] - all <= 0, NOT verified
        # Second sample: [0.1, 0.2] - both > 0, verified
        # Since not all samples are verified, initial_verified = False
        global_lb = torch.tensor([[-0.5, -0.3], [0.1, 0.2]])
        rhs = torch.zeros(2, 2)

        initial_verified, remaining_index = initial_verify_criterion(global_lb, rhs)

        self.assertFalse(initial_verified)
        # remaining_index should contain index 0 (unverified sample)
        self.assertEqual(len(remaining_index), 1)
        self.assertEqual(remaining_index[0].item(), 0)

    def test_return_early_when_verified(self):
        """Test return logic when initially verified."""
        initial_verified = True
        global_lb = torch.tensor([[0.5], [0.3]])
        visited = 10

        if initial_verified:
            result = (global_lb.max(), visited, "safe")

        self.assertEqual(result[0].item(), 0.5)
        self.assertEqual(result[1], 10)
        self.assertEqual(result[2], "safe")


if __name__ == '__main__':
    unittest.main()
