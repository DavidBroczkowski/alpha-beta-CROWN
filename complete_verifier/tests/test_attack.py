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
"""Unit tests for attack package."""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import math
import tempfile

import torch
import torch.nn as nn
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize arguments.Config before importing attack modules
import arguments
arguments.Config.parse_config(args=[], verbose=False)

from auto_LiRPA import BoundedTensor, BoundedModule
from auto_LiRPA.perturbations import PerturbationLpNorm

from attack.attack_utils import (
    Stats, AdamClipping, PGDAttackResult,
    precompute_group_indices, process_data_for_attack,
    boundary_attack, default_adv_verifier, is_specification_vio
)
from attack.domains import ReLUDomain, SortedReLUDomainList, to_cpu, to_device
from attack.adv_domains import AdvExample, AdvExamplePool


class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestAdamClipping(unittest.TestCase):
    """Tests for AdamClipping optimizer."""

    def test_init_valid_params(self):
        """Test AdamClipping initialization with valid parameters."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)

    def test_init_invalid_lr_raises(self):
        """Test that invalid learning rate raises ValueError."""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamClipping([param], lr=-0.01)

    def test_init_invalid_eps_raises(self):
        """Test that invalid epsilon raises ValueError."""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamClipping([param], eps=-1e-8)

    def test_init_invalid_beta0_raises(self):
        """Test that invalid beta[0] raises ValueError."""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamClipping([param], betas=(1.5, 0.999))

    def test_init_invalid_beta1_raises(self):
        """Test that invalid beta[1] raises ValueError."""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamClipping([param], betas=(0.9, 1.5))

    def test_init_invalid_weight_decay_raises(self):
        """Test that invalid weight_decay raises ValueError."""
        param = torch.randn(10, requires_grad=True)
        with self.assertRaises(ValueError):
            AdamClipping([param], weight_decay=-0.1)

    def test_step_without_clipping(self):
        """Test step without clipping."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)

        # Simulate a backward pass
        loss = param.sum()
        loss.backward()

        # Take a step
        optimizer.step()

        # Parameter should have been updated
        self.assertTrue(param.grad is not None)

    def test_step_with_clipping(self):
        """Test step with clipping."""
        param = torch.full((10,), 0.5, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.1)

        lower_limit = torch.zeros(10)
        upper_limit = torch.ones(10)

        # Simulate a backward pass
        loss = param.sum()
        loss.backward()

        # Take a step with clipping
        optimizer.step(clipping=True, lower_limit=lower_limit,
                       upper_limit=upper_limit, sign=1)

        # Parameter should be within bounds
        self.assertTrue((param >= lower_limit).all())
        self.assertTrue((param <= upper_limit).all())

    def test_setstate(self):
        """Test __setstate__ sets amsgrad default."""
        param = torch.randn(10, requires_grad=True)
        optimizer = AdamClipping([param], lr=0.01)

        # Get state and restore it
        state = optimizer.__getstate__()
        optimizer.__setstate__(state)

        # Check amsgrad is set
        self.assertIn('amsgrad', optimizer.param_groups[0])

    def test_step_with_amsgrad(self):
        """Test step with AMSGrad variant."""
        param = torch.randn(10, requires_grad=True)
        param_before = param.clone().detach()
        optimizer = AdamClipping([param], lr=0.01, amsgrad=True)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Verify parameter was updated (not equal to original)
        self.assertFalse(torch.equal(param.data, param_before),
            "Parameter should be updated after optimizer step with amsgrad")

        # Verify amsgrad state is maintained
        self.assertTrue(optimizer.param_groups[0]['amsgrad'],
            "amsgrad should be enabled in optimizer")

    def test_step_with_weight_decay(self):
        """Test step with weight decay."""
        param = torch.randn(10, requires_grad=True)
        param_before = param.clone().detach()
        optimizer = AdamClipping([param], lr=0.01, weight_decay=0.01)

        loss = param.sum()
        loss.backward()
        optimizer.step()

        # Verify parameter was updated (not equal to original)
        self.assertFalse(torch.equal(param.data, param_before),
            "Parameter should be updated after optimizer step with weight_decay")

        # Verify weight_decay is set in optimizer
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], 0.01,
            "weight_decay should be set to 0.01")


class TestPrecomputeGroupIndices(unittest.TestCase):
    """Tests for precompute_group_indices function."""

    def test_uniform_groups(self):
        """Test with uniform group sizes."""
        or_spec_size = torch.tensor([3, 3, 3])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 3)
        self.assertEqual(group_indices.shape, (9,))
        expected = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertTrue(torch.equal(group_indices, expected))

    def test_varying_groups(self):
        """Test with varying group sizes."""
        or_spec_size = torch.tensor([2, 3, 1])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 3)
        self.assertEqual(group_indices.shape, (6,))
        expected = torch.tensor([0, 0, 1, 1, 1, 2])
        self.assertTrue(torch.equal(group_indices, expected))

    def test_single_group(self):
        """Test with single group."""
        or_spec_size = torch.tensor([5])
        group_indices, num_or = precompute_group_indices(or_spec_size)

        self.assertEqual(num_or, 1)
        self.assertEqual(group_indices.shape, (5,))
        expected = torch.tensor([0, 0, 0, 0, 0])
        self.assertTrue(torch.equal(group_indices, expected))


class TestProcessDataForAttack(unittest.TestCase):
    """Tests for process_data_for_attack function."""

    def test_expand_single_input(self):
        """Test expanding single input to multiple OR specs."""
        # Create bounded tensor with batch size 1
        data = torch.randn(1, 10)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        c = torch.randn(1, 6, 5)  # 6 specs total
        rhs = torch.randn(1, 6)
        or_spec_size = torch.tensor([2, 2, 2])  # 3 OR clauses, 2 AND each

        x_out, data_min, data_max, c_out, rhs_out = process_data_for_attack(
            x, c, rhs, or_spec_size
        )

        self.assertEqual(x_out.shape[0], 3)  # Expanded to num_or
        self.assertEqual(c_out.shape, (6, 5))  # Flattened specs
        self.assertEqual(rhs_out.shape, (6,))

    def test_matching_batch_and_or_size(self):
        """Test when batch size matches num_or."""
        num_or = 3
        data = torch.randn(num_or, 10)
        ptb = PerturbationLpNorm(norm=float('inf'), x_L=data - 0.1, x_U=data + 0.1)
        x = BoundedTensor(data, ptb)

        c = torch.randn(num_or, 2, 5)  # 2 AND per OR
        rhs = torch.randn(num_or, 2)
        or_spec_size = torch.tensor([2, 2, 2])

        x_out, data_min, data_max, c_out, rhs_out = process_data_for_attack(
            x, c, rhs, or_spec_size
        )

        self.assertEqual(x_out.shape[0], num_or)
        self.assertEqual(c_out.shape, (6, 5))


class TestBoundaryAttack(unittest.TestCase):
    """Tests for boundary_attack function."""

    def test_small_perturbation(self):
        """Test boundary attack with small number of perturbed pixels."""
        model = SimpleModel(input_size=10)
        x = torch.randn(1, 10)
        data_min = x.clone()
        data_max = x.clone()
        # Perturb only 3 pixels
        data_min[0, :3] -= 0.1
        data_max[0, :3] += 0.1

        result = boundary_attack(model, x, data_min, data_max)

        self.assertIsNotNone(result)
        # Should generate 2^3 = 8 adversarial inputs
        self.assertEqual(result.shape[0], 8)

    def test_large_perturbation_returns_none(self):
        """Test boundary attack returns None for large perturbation."""
        model = SimpleModel(input_size=10)
        x = torch.randn(1, 10)
        data_min = x - 0.1  # All pixels perturbed
        data_max = x + 0.1

        result = boundary_attack(model, x, data_min, data_max)

        self.assertIsNone(result)


class TestDefaultAdvVerifier(unittest.TestCase):
    """Tests for default_adv_verifier function."""

    def test_without_vnnlib_returns_true(self):
        """Test verifier returns True without vnnlib."""
        adv_input = torch.randn(1, 10)
        adv_output = torch.randn(1, 5)

        result = default_adv_verifier(adv_input, adv_output, vnnlib=None)

        self.assertTrue(result)


class TestIsSpecificationVio(unittest.TestCase):
    """Tests for is_specification_vio function."""

    def test_specification_violated(self):
        """Test when specification is violated."""
        # Create a simple spec: y[0] <= 0
        input_box = [(0.0, 1.0)] * 5
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [0.0]
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)
        expected_y = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0])  # y[0] = -0.5 < 0, violates y[0] <= 0? No, satisfies.

        # Actually, is_specification_vio checks if ANY spec in spec_list is SAT (violated from verifier perspective)
        # If vec = mat @ y <= rhs + tol, then sat=True, meaning the spec is violated (adversarial found)
        result = is_specification_vio(box_spec_list, x_list, expected_y, tol=1e-4)

        self.assertTrue(result)  # Spec is satisfied (attacked successfully)

    def test_specification_not_violated(self):
        """Test when specification is not violated."""
        input_box = [(0.0, 1.0)] * 5
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [-1.0]  # y[0] <= -1.0
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)
        expected_y = torch.tensor([0.5, 0.0, 0.0, 0.0, 0.0])  # y[0] = 0.5 > -1.0, does not satisfy

        result = is_specification_vio(box_spec_list, x_list, expected_y, tol=1e-4)

        self.assertFalse(result)

    def test_input_outside_box(self):
        """Test when input is outside the box."""
        input_box = [(0.0, 0.1)] * 5  # Very small box
        prop_mat = [[1.0, 0.0, 0.0, 0.0, 0.0]]
        prop_rhs = [0.0]
        spec_list = [(prop_mat, prop_rhs)]
        box_spec_list = [(input_box, spec_list)]

        x_list = torch.tensor([0.5] * 5)  # Outside the box
        expected_y = torch.tensor([-0.5, 0.0, 0.0, 0.0, 0.0])

        result = is_specification_vio(box_spec_list, x_list, expected_y, tol=1e-4)

        self.assertFalse(result)  # Input outside box, so no violation found


class TestAdvExample(unittest.TestCase):
    """Tests for AdvExample class."""

    def test_init(self):
        """Test AdvExample initialization."""
        x = torch.randn(1, 10)
        obj = -0.5
        pattern = [torch.randint(0, 2, (20,))]

        adv = AdvExample(x, obj, pattern)

        self.assertTrue(torch.equal(adv.x, x))
        self.assertEqual(adv.obj, obj)
        self.assertEqual(len(adv.activation_pattern), 1)

    def test_lt(self):
        """Test __lt__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -0.5, [])

        # adv1 has smaller obj (-1.0 < -0.5)
        self.assertTrue(adv1 < adv2)

    def test_le(self):
        """Test __le__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -1.0, [])

        self.assertTrue(adv1 <= adv2)

    def test_eq(self):
        """Test __eq__ comparison."""
        adv1 = AdvExample(torch.randn(1, 10), -1.0, [])
        adv2 = AdvExample(torch.randn(1, 10), -1.0, [])

        self.assertTrue(adv1 == adv2)


class TestAdvExamplePool(unittest.TestCase):
    """Tests for AdvExamplePool class."""

    def _create_bounded_model(self):
        """Create a bounded model for testing."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        return bounded_model

    def test_init(self):
        """Test AdvExamplePool initialization."""
        model = self._create_bounded_model()
        # Create unstable masks for each ReLU layer
        unstable_mask = [torch.ones(1, 20)]  # One ReLU layer with 20 neurons

        pool = AdvExamplePool(model, unstable_mask, capacity=100)

        self.assertEqual(pool.capacity, 100)
        self.assertEqual(len(pool.adv_pool), 0)

    def test_get_var_empty_pool(self):
        """Test get_var with empty pool."""
        model = self._create_bounded_model()
        unstable_mask = [torch.ones(1, 20)]

        pool = AdvExamplePool(model, unstable_mask)
        result = pool.get_var()

        self.assertIsNone(result)

    def test_replace_adv_example_add(self):
        """Test replace_adv_example adds new example."""
        model = self._create_bounded_model()
        unstable_mask = [torch.ones(1, 20)]

        pool = AdvExamplePool(model, unstable_mask)

        adv = AdvExample(torch.randn(10), -0.5, [torch.randint(0, 2, (20,))])
        c_replaced, c_added, c_rejected = pool.replace_adv_example(adv, 0, 0, 0)

        self.assertEqual(c_added, 1)
        self.assertEqual(len(pool.adv_pool), 1)


class TestAdvExamplePoolActivationPattern(unittest.TestCase):
    """Tests for AdvExamplePool activation pattern methods."""

    def _create_pool_with_examples(self):
        """Create a pool with some adversarial examples."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        unstable_mask = [torch.ones(20)]  # One ReLU layer

        pool = AdvExamplePool(bounded_model, unstable_mask, capacity=100)

        # Add some examples manually
        for i in range(5):
            pattern = [torch.randint(0, 2, (20,))]
            adv = AdvExample(torch.randn(10), -float(i) / 10, pattern)
            pool.adv_pool.add(adv)

        return pool

    def test_get_activation_pattern_from_pool(self):
        """Test getting common activation patterns."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_activation_pattern_from_pool(prob_threshold=0.6)

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_ranked_activation_pattern(self):
        """Test getting ranked activation patterns."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_ranked_activation_pattern()

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_ranked_activation_pattern_with_limit(self):
        """Test getting ranked activation patterns with n_activations limit."""
        pool = self._create_pool_with_examples()

        decisions, coeffs = pool.get_ranked_activation_pattern(n_activations=5)

        self.assertEqual(len(decisions), 5)
        self.assertEqual(len(coeffs), 5)

    def test_get_activation_pattern(self):
        """Test getting activation pattern from single example."""
        pool = self._create_pool_with_examples()
        adv_example = pool.adv_pool[0]

        decisions, coeffs = pool.get_activation_pattern(adv_example)

        self.assertIsInstance(decisions, torch.Tensor)
        self.assertIsInstance(coeffs, torch.Tensor)

    def test_get_activation_pattern_with_blacklist(self):
        """Test getting activation pattern with blacklist."""
        pool = self._create_pool_with_examples()
        adv_example = pool.adv_pool[0]

        # Blacklist some neurons
        blacklist = [[0, 0], [0, 1], [0, 2]]

        decisions, coeffs = pool.get_activation_pattern(adv_example, blacklist=blacklist)

        # Blacklisted neurons should not appear in decisions
        for d in decisions.tolist():
            self.assertNotIn(d, blacklist)


class TestAdvExamplePoolMostLikelyActivation(unittest.TestCase):
    """Tests for find_most_likely_activation method."""

    def test_find_most_likely_activation(self):
        """Test finding most likely activation for given decisions."""
        model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
        dummy_input = torch.randn(1, 10)
        bounded_model = BoundedModule(model, dummy_input)
        unstable_mask = [torch.ones(20)]

        pool = AdvExamplePool(bounded_model, unstable_mask)

        # Add examples with consistent patterns
        for i in range(5):
            pattern = [torch.ones(20)]  # All active
            adv = AdvExample(torch.randn(10), -float(i) / 10, pattern)
            pool.adv_pool.add(adv)

        # Get decisions that exist
        decisions, _ = pool.get_activation_pattern_from_pool(prob_threshold=0.5)

        if len(decisions) > 0:
            test_decisions = decisions[:2].tolist()
            coeffs = pool.find_most_likely_activation(test_decisions)
            self.assertEqual(len(coeffs), len(test_decisions))


if __name__ == '__main__':
    unittest.main()
