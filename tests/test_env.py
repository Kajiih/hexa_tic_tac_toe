import pytest
from pettingzoo.test import api_test

from hexa_tic_tac_toe.env import env


def test_pettingzoo_api_compliance() -> None:
    """Tests that the custom environment complies with the standard PettingZoo AEC API."""
    # Use a small radius for fast testing
    test_env = env(radius=10)
    
    # Run the strict PettingZoo API validation test
    api_test(test_env, num_cycles=1000, verbose_progress=False)
