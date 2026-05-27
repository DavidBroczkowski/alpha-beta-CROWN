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
"""Unit tests for custom package modules."""
import os
import sys
import unittest

import torch
import torch.nn as nn
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from custom.custom_model_loader import (
    conv_output_shape, transpose_linear_layers
)
from custom.custom_op import (
    LinearMaskedReluOp, LinearMaskedRelu
)


class TestConvOutputShapeCustom(unittest.TestCase):
    """Tests for conv_output_shape function in custom_model_loader."""

    def test_basic_conv(self):
        """Test basic convolution output shape."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 30)

    def test_same_padding(self):
        """Test convolution with same padding."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=1)
        self.assertEqual(h, 32)
        self.assertEqual(w, 32)

    def test_stride_2(self):
        """Test convolution with stride 2."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=2, pad=1)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_int_input(self):
        """Test with integer input instead of tuple."""
        h, w = conv_output_shape(h_w=32, kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 30)

    def test_dilation(self):
        """Test convolution with dilation."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=0, dilation=2)
        self.assertEqual(h, 28)
        self.assertEqual(w, 28)

    def test_asymmetric_input(self):
        """Test convolution with asymmetric input."""
        h, w = conv_output_shape(h_w=(28, 32), kernel_size=3, stride=1, pad=0)
        self.assertEqual(h, 26)
        self.assertEqual(w, 30)

    def test_1x1_conv(self):
        """Test 1x1 convolution."""
        h, w = conv_output_shape(h_w=(16, 16), kernel_size=1, stride=1, pad=0)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)


class TestTransposeLinearLayers(unittest.TestCase):
    """Tests for transpose_linear_layers function."""

    def test_transpose_single_linear(self):
        """Test transposing a single linear layer."""
        model = nn.Sequential(
            nn.Linear(10, 20)
        )
        model[0].weight.data = torch.randn(20, 10)
        model[0].bias.data = torch.randn(20)

        result = transpose_linear_layers(model)

        self.assertIsInstance(result, nn.Sequential)
        self.assertEqual(result[0].in_features, 10)
        self.assertEqual(result[0].out_features, 20)

    def test_transpose_multiple_linear(self):
        """Test transposing multiple linear layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        result = transpose_linear_layers(model)

        self.assertIsInstance(result, nn.Sequential)
        self.assertEqual(len(result), 3)
        self.assertIsInstance(result[0], nn.Linear)
        self.assertIsInstance(result[1], nn.ReLU)
        self.assertIsInstance(result[2], nn.Linear)

    def test_transpose_preserves_activation(self):
        """Test that activation layers are preserved."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Sigmoid()
        )

        result = transpose_linear_layers(model)

        self.assertIsInstance(result[1], nn.ReLU)
        self.assertIsInstance(result[3], nn.Sigmoid)

    def test_transpose_with_no_bias(self):
        """Test transposing linear layer without bias."""
        model = nn.Sequential(
            nn.Linear(10, 20, bias=False)
        )
        model[0].bias = None

        result = transpose_linear_layers(model)

        self.assertIsInstance(result[0], nn.Linear)


class TestLinearMaskedReluOp(unittest.TestCase):
    """Tests for LinearMaskedReluOp custom autograd function."""

    def test_forward_all_relu(self):
        """Test forward pass with all ReLU (mask=0)."""
        input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        mask = torch.zeros(4)
        slope = torch.ones(4)
        bias = torch.zeros(4)

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        expected = torch.tensor([0.0, 0.0, 1.0, 2.0])  # ReLU output
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_all_linear(self):
        """Test forward pass with all linear (mask=1)."""
        input_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        mask = torch.ones(4)
        slope = torch.tensor([2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([1.0, 1.0, 1.0, 1.0])

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        expected = torch.tensor([-1.0, 1.0, 3.0, 5.0])  # 2*x + 1
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_mixed(self):
        """Test forward pass with mixed mask."""
        input_tensor = torch.tensor([-1.0, -1.0, 1.0, 1.0])
        mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
        slope = torch.tensor([2.0, 2.0, 2.0, 2.0])
        bias = torch.tensor([0.5, 0.5, 0.5, 0.5])

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)

        # mask=0: ReLU(-1)=0, ReLU(1)=1
        # mask=1: 2*(-1)+0.5=-1.5, 2*1+0.5=2.5
        expected = torch.tensor([0.0, -1.5, 1.0, 2.5])
        self.assertTrue(torch.allclose(result, expected))

    def test_forward_gradient_enabled(self):
        """Test that gradients can be computed."""
        input_tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        mask = torch.zeros(2)
        slope = torch.ones(2)
        bias = torch.zeros(2)

        result = LinearMaskedReluOp.apply(input_tensor, mask, slope, bias)
        loss = result.sum()
        loss.backward()

        self.assertIsNotNone(input_tensor.grad)


class TestLinearMaskedRelu(unittest.TestCase):
    """Tests for LinearMaskedRelu module."""

    def test_init_int_size(self):
        """Test initialization with integer size."""
        module = LinearMaskedRelu(10)

        self.assertEqual(module.mask.shape, (10,))
        self.assertEqual(module.slope.shape, (10,))
        self.assertEqual(module.bias.shape, (10,))

    def test_init_tuple_size(self):
        """Test initialization with tuple size."""
        module = LinearMaskedRelu((3, 4, 5))

        self.assertEqual(module.mask.shape, (3, 4, 5))
        self.assertEqual(module.slope.shape, (3, 4, 5))
        self.assertEqual(module.bias.shape, (3, 4, 5))

    def test_forward_shape(self):
        """Test that forward preserves shape."""
        module = LinearMaskedRelu(10)
        input_tensor = torch.randn(5, 10)

        result = module(input_tensor)

        self.assertEqual(result.shape, (5, 10))

    def test_forward_deterministic(self):
        """Test that forward is deterministic."""
        module = LinearMaskedRelu(10)
        input_tensor = torch.randn(5, 10)

        result1 = module(input_tensor)
        result2 = module(input_tensor)

        self.assertTrue(torch.allclose(result1, result2))

    def test_buffers_registered(self):
        """Test that mask, slope, bias are registered as buffers."""
        module = LinearMaskedRelu(10)

        buffer_names = [name for name, _ in module.named_buffers()]
        self.assertIn('mask', buffer_names)
        self.assertIn('slope', buffer_names)
        self.assertIn('bias', buffer_names)

    def test_no_parameters(self):
        """Test that module has no learnable parameters."""
        module = LinearMaskedRelu(10)

        params = list(module.parameters())
        self.assertEqual(len(params), 0)


class TestCustomOpIntegration(unittest.TestCase):
    """Integration tests for custom operations."""

    def test_linear_masked_relu_in_sequential(self):
        """Test LinearMaskedRelu in a sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            LinearMaskedRelu(20),
            nn.Linear(20, 5)
        )

        input_tensor = torch.randn(3, 10)
        result = model(input_tensor)

        self.assertEqual(result.shape, (3, 5))

    def test_linear_masked_relu_conv_shapes(self):
        """Test LinearMaskedRelu with conv layer output shapes."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            LinearMaskedRelu((16, 32, 32)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1)
        )

        input_tensor = torch.randn(2, 3, 32, 32)
        result = model(input_tensor)

        self.assertEqual(result.shape, (2, 32, 32, 32))

    def test_backward_pass(self):
        """Test that backward pass works through LinearMaskedRelu."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            LinearMaskedRelu(20),
            nn.Linear(20, 5)
        )

        input_tensor = torch.randn(3, 10, requires_grad=True)
        result = model(input_tensor)
        loss = result.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(model[0].weight.grad)
        self.assertIsNotNone(model[2].weight.grad)


class TestConvOutputShapeEdgeCases(unittest.TestCase):
    """Edge case tests for conv_output_shape."""

    def test_large_kernel(self):
        """Test with kernel larger than input."""
        h, w = conv_output_shape(h_w=(5, 5), kernel_size=7, stride=1, pad=3)
        self.assertEqual(h, 5)
        self.assertEqual(w, 5)

    def test_large_stride(self):
        """Test with large stride."""
        h, w = conv_output_shape(h_w=(64, 64), kernel_size=7, stride=4, pad=3)
        self.assertEqual(h, 16)
        self.assertEqual(w, 16)

    def test_asymmetric_kernel_tuple(self):
        """Test with asymmetric kernel as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=(3, 5), stride=1, pad=0)
        self.assertEqual(h, 30)
        self.assertEqual(w, 28)

    def test_asymmetric_stride_tuple(self):
        """Test with asymmetric stride as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=(2, 1), pad=1)
        self.assertEqual(h, 16)
        self.assertEqual(w, 32)

    def test_asymmetric_pad_tuple(self):
        """Test with asymmetric padding as tuple."""
        h, w = conv_output_shape(h_w=(32, 32), kernel_size=3, stride=1, pad=(0, 1))
        self.assertEqual(h, 30)
        self.assertEqual(w, 32)


if __name__ == '__main__':
    unittest.main()
