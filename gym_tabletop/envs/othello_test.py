import unittest

import numpy as np

from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.othello import OthelloEnv


class TestOthelloEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = OthelloEnv()

    def test_get_possible_actions(self):
        self.env.reset()
        expected = {(2, 3), (3, 2), (4, 5), (5, 4)}
        self.assertEqual(set(self.env.get_available_actions()), expected)

        self.env.current_player = 2
        expected = {(2, 4), (3, 5), (4, 2), (5, 3)}
        self.assertEqual(set(self.env.get_available_actions()), expected)

        self.env.reset()
        self.env.board[[3, 3, 4], [2, 3, 3]] = 1
        self.env.board[[2, 3, 4], [4, 4, 4]] = 2
        self.env._compute_edge_set()
        expected = {(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)}
        self.assertEqual(set(self.env.get_available_actions()), expected)

        self.env.reset()
        self.env.board = np.ones((8, 8))
        self.env.board[[3, 4, 4, 5, 6], [7, 6, 7, 6, 7]] = 0
        self.env.board[5, 7] = 2
        self.env._compute_edge_set()
        self.assertEqual(self.env.get_available_actions(), [])

        self.env.current_player = 2
        self.assertEqual(self.env.get_available_actions(), [])

