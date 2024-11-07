import unittest
from HMM import *


class TestHMM(unittest.TestCase):
    def setUp(self):
        self.hmm = HMM()
        self.hmm.load("cat")

    def test_load_transitions(self):
        expected_transitions = {
            '#': {'happy': 0.5, 'grumpy': 0.5, 'hungry': 0.0},
            'happy': {'happy': 0.5, 'grumpy': 0.1, 'hungry': 0.4},
            'grumpy': {'happy': 0.6, 'grumpy': 0.3, 'hungry': 0.1},
            'hungry': {'happy': 0.1, 'grumpy': 0.6, 'hungry': 0.3}
        }
        print(self.hmm.transitions)
        self.assertEqual(self.hmm.transitions, expected_transitions, "Transitions do not match expected values")

    def test_load_emissions(self):
        expected_emissions = {
            'happy': {'silent': 0.2, 'meow': 0.3, 'purr': 0.5},
            'grumpy': {'silent': 0.5, 'meow': 0.4, 'purr': 0.1},
            'hungry': {'silent': 0.2, 'meow': 0.6, 'purr': 0.2}
        }
        self.assertEqual(self.hmm.emissions, expected_emissions, "Emissions do not match expected values")


if __name__ == "__main__":
    unittest.main()
