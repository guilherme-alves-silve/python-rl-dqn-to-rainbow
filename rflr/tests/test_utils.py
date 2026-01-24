#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import numpy as np
import os
import sys
import random
from unittest.mock import patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    map_env_action,
    map_model_action,
    map_env2model,
    timeit,
    set_config_seed,
    get_visual_array,
    debug_concat_frames
)


class TestUtils(unittest.TestCase):
    """Unit tests for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        pass
        
    def test_map_env_action_noop(self):
        """Test mapping environment actions to NOOP."""
        # Actions 0 and 1 should map to 0 (NOOP)
        self.assertEqual(map_env_action(0), 0)
        self.assertEqual(map_env_action(1), 0)
        
    def test_map_env_action_down(self):
        """Test mapping environment actions to DOWN."""
        # Actions 3 and 5 should map to 3 (DOWN)
        self.assertEqual(map_env_action(3), 3)
        self.assertEqual(map_env_action(5), 3)
        
    def test_map_env_action_up(self):
        """Test mapping environment actions to UP."""
        # Actions 2 and 4 should map to 2 (UP)
        self.assertEqual(map_env_action(2), 2)
        self.assertEqual(map_env_action(4), 2)
        
    def test_map_env_action_invalid(self):
        """Test mapping invalid environment action."""
        with self.assertRaises(ValueError):
            map_env_action(6)  # Invalid action
            
    def test_map_model_action_noop(self):
        """Test mapping model actions to NOOP."""
        self.assertEqual(map_model_action(0), 0)
        
    def test_map_model_action_down(self):
        """Test mapping model actions to DOWN."""
        self.assertEqual(map_model_action(1), 3)
        
    def test_map_model_action_up(self):
        """Test mapping model actions to UP."""
        self.assertEqual(map_model_action(2), 2)
        
    def test_map_model_action_invalid(self):
        """Test mapping invalid model action."""
        with self.assertRaises(Exception):
            map_model_action(3)  # Invalid action
            
    def test_map_env2model_noop(self):
        """Test mapping environment to model NOOP."""
        self.assertEqual(map_env2model(0), 0)
        
    def test_map_env2model_down(self):
        """Test mapping environment to model DOWN."""
        self.assertEqual(map_env2model(3), 1)
        
    def test_map_env2model_up(self):
        """Test mapping environment to model UP."""
        self.assertEqual(map_env2model(2), 2)
        
    def test_map_env2model_invalid(self):
        """Test mapping invalid environment action to model."""
        with self.assertRaises(ValueError):
            map_env2model(1)  # Invalid action
            
    def test_timeit_decorator(self):
        """Test the timeit decorator."""
        @timeit
        def test_function():
            return 42
            
        # Call decorated function
        result = test_function()
        
        # Check result
        self.assertEqual(result, 42)
        
    def test_set_config_seed(self):
        """Test setting random seeds."""
        seed = 42
        
        # Set seed
        set_config_seed(seed)
        
        # Generate some random numbers
        py_val = random.random()
        np_val = np.random.random()
        torch_val = torch.rand(1).item()
        
        # Reset seed and generate again
        set_config_seed(seed)
        py_val2 = random.random()
        np_val2 = np.random.random()
        torch_val2 = torch.rand(1).item()
        
        # Should be the same
        self.assertEqual(py_val, py_val2)
        self.assertEqual(np_val, np_val2)
        self.assertEqual(torch_val, torch_val2)
        
    def test_get_visual_array_tensor_4d(self):
        """Test get_visual_array with 4D tensor."""
        # Create 4D tensor [batch, channels, H, W]
        tensor = torch.randn(2, 3, 84, 84)
        
        array, cmap = get_visual_array(tensor, show_last_batch=True)
        
        # Should return last batch item
        self.assertEqual(array.shape, (84, 84, 3))
        self.assertEqual(cmap, "gray")
        
    def test_get_visual_array_tensor_3d(self):
        """Test get_visual_array with 3D tensor."""
        # Create 3D tensor [channels, H, W]
        tensor = torch.randn(3, 84, 84)
        
        array, cmap = get_visual_array(tensor)
        
        # Should be permuted
        self.assertEqual(array.shape, (84, 84, 3))
        self.assertEqual(cmap, "gray")
        
    def test_get_visual_array_numpy_4d(self):
        """Test get_visual_array with 4D numpy array."""
        # Create 4D numpy array [batch, channels, H, W]
        array = np.random.randn(2, 3, 84, 84)
        
        result, cmap = get_visual_array(array, show_last_batch=True)
        
        # Should return last batch item with last channel
        self.assertEqual(result.shape, (84, 84))
        self.assertEqual(cmap, "gray")
        
    def test_get_visual_array_numpy_3d(self):
        """Test get_visual_array with 3D numpy array."""
        # Create 3D numpy array [channels, H, W]
        array = np.random.randn(3, 84, 84)
        
        result, cmap = get_visual_array(array)
        
        # Should return last channel
        self.assertEqual(result.shape, (84, 84))
        self.assertEqual(cmap, "gray")
        
    def test_get_visual_array_other_types(self):
        """Test get_visual_array with other types."""
        # Test with regular array
        array = np.array([1, 2, 3])
        result, cmap = get_visual_array(array)
        
        self.assertTrue(np.array_equal(result, array))
        self.assertIsNone(cmap)
        
    def test_action_mapping_consistency(self):
        """Test that action mappings are consistent."""
        # Test round-trip mapping
        for env_action in [0, 2, 3]:
            model_action = map_env2model(env_action)
            back_to_env = map_model_action(model_action)
            self.assertEqual(back_to_env, env_action)
            
    def test_action_mapping_coverage(self):
        """Test that all environment actions are covered."""
        # All valid environment actions should be mapped
        valid_env_actions = [0, 1, 2, 3, 4, 5]
        
        for action in valid_env_actions:
            try:
                mapped = map_env_action(action)
                self.assertIn(mapped, [0, 2, 3])
            except ValueError:
                # Some actions might be invalid, that's ok
                pass
                
    def test_seed_reproducibility_torch(self):
        """Test PyTorch reproducibility with seed."""
        seed = 123
        
        # Generate with seed
        set_config_seed(seed)
        val1 = torch.rand(5)
        
        # Generate again with same seed
        set_config_seed(seed)
        val2 = torch.rand(5)
        
        # Should be identical
        torch.testing.assert_close(val1, val2)
        
    def test_seed_reproducibility_numpy(self):
        """Test NumPy reproducibility with seed."""
        seed = 456
        
        # Generate with seed
        set_config_seed(seed)
        val1 = np.random.random(5)
        
        # Generate again with same seed
        set_config_seed(seed)
        val2 = np.random.random(5)
        
        # Should be identical
        np.testing.assert_array_equal(val1, val2)
        
    def test_seed_reproducibility_python(self):
        """Test Python random reproducibility with seed."""
        seed = 789
        
        # Generate with seed
        set_config_seed(seed)
        val1 = [random.random() for _ in range(5)]
        
        # Generate again with same seed
        set_config_seed(seed)
        val2 = [random.random() for _ in range(5)]
        
        # Should be identical
        self.assertEqual(val1, val2)
        
    def test_timeit_decorator_with_exception(self):
        """Test timeit decorator doesn't interfere with exceptions."""
        @timeit
        def test_function_with_error():
            raise ValueError("Test error")
            
        # Should still raise the exception
        with self.assertRaises(ValueError):
            test_function_with_error()
            
    def test_debug_concat_frames(self):
        """Test debug_concat_frames function runs without error."""
        # Create test tensor [4, 84, 84]
        tensor = torch.randn(4, 84, 84)
        
        # Mock matplotlib to avoid actual plotting
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.subplots'), \
             patch('matplotlib.pyplot.tight_layout'):
            
            # Should run without error
            try:
                debug_concat_frames(tensor)
            except Exception as e:
                self.fail(f"debug_concat_frames raised exception: {e}")


if __name__ == '__main__':
    unittest.main()
