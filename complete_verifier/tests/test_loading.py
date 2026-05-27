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
"""Unit tests for loading.py module."""

import os
import sys
import pytest
import tempfile
import torch
import numpy as np

# Add complete_verifier to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# joint_optimization_with_onnx_and_vnnlib Tests
# ============================================================================

class TestJointOptimizationWithOnnxAndVnnlib:
    """Tests for joint_optimization_with_onnx_and_vnnlib function."""

    def test_none_flag(self):
        """Test with 'none' flag - should not modify anything."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        # Create a mock model
        class MockModel:
            output_names = ['output']
            onnx_model = None

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]
        flags = ['none']

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        assert result_model is model
        assert result_vnnlib == vnnlib

    def test_empty_flags(self):
        """Test with empty flags list."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        class MockModel:
            output_names = ['output']
            onnx_model = None

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]
        flags = []

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        assert result_model is model
        assert result_vnnlib == vnnlib


# ============================================================================
# load_verification_dataset Tests
# ============================================================================

class TestLoadVerificationDataset:
    """Tests for load_verification_dataset function."""

    def test_unsupported_dataset_raises(self):
        """Test that unsupported dataset raises NotImplementedError."""
        import arguments
        from loading import load_verification_dataset

        arguments.Config['data']['dataset'] = 'UNSUPPORTED_DATASET'
        arguments.Config['data']['pkl_path'] = None

        with pytest.raises(NotImplementedError, match='Dataset not supported'):
            load_verification_dataset()


class TestJointOptimizationAdditional:
    """Additional tests for joint_optimization_with_onnx_and_vnnlib."""

    def test_multiple_none_flags(self):
        """Test with multiple 'none' flags."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        class MockModel:
            output_names = ['output']
            onnx_model = None

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]
        flags = ['none', 'none', 'none']

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        assert result_model is model
        assert result_vnnlib == vnnlib

    def test_peel_off_sigmoid_flag_no_sigmoid(self):
        """Test peel_off_last_softmax_layer flag when no sigmoid layer exists."""
        from loading import joint_optimization_with_onnx_and_vnnlib

        class MockOnnxGraph:
            node = []  # No nodes

        class MockOnnxModel:
            graph = MockOnnxGraph()

        class MockModel:
            output_names = ['output']
            onnx_model = MockOnnxModel()

        model = MockModel()
        shape = [-1, 3, 32, 32]
        vnnlib = [('input', [(np.array([[1, -1]]), np.array([0.0]))])]
        flags = ['peel_off_last_softmax_layer']

        result_model, result_vnnlib = joint_optimization_with_onnx_and_vnnlib(
            model, shape, vnnlib, flags)

        # No sigmoid found, so model and vnnlib should be unchanged
        assert result_model is model
        assert result_vnnlib == vnnlib


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
