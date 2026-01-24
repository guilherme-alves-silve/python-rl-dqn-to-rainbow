#!/usr/bin/env python
# coding: utf-8

"""
Test runner for DQN implementation

Run all unit tests or specific test modules.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_network import TestNetwork
from tests.test_replay_buffer import TestReplayBuffer, TestBinaryReplayBuffer
from tests.test_agent import TestAgentDQN
from tests.test_preprocessing import TestPreprocessingWrapper
from tests.test_utils import TestUtils


def run_all_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestNetwork,
        TestReplayBuffer,
        TestBinaryReplayBuffer,
        TestAgentDQN,
        TestPreprocessingWrapper,
        TestUtils
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_specific_test(test_class):
    """Run a specific test class."""
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    """Main function to run tests."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1].lower()
        
        test_mapping = {
            'network': TestNetwork,
            'replay': TestReplayBuffer,
            'binary': TestBinaryReplayBuffer,
            'agent': TestAgentDQN,
            'preprocessing': TestPreprocessingWrapper,
            'utils': TestUtils
        }
        
        if test_name in test_mapping:
            success = run_specific_test(test_mapping[test_name])
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: network, replay, binary, agent, preprocessing, utils")
            print("Or run with no arguments to run all tests")
            sys.exit(1)
    else:
        # Run all tests
        print("Running all tests...")
        print("=" * 60)
        success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
