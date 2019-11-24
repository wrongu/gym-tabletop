import unittest

from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.connect4 import ConnectFourEnv


class TestConnectFourEnv(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ConnectFourEnv()

    def test_get_possible_actions(self):
        expected = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(self.env.get_available_actions(), expected)

        self.env.offsets[:] = 0
        self.assertEqual(self.env.get_available_actions(), [])

    def test_evaluate_state(self):
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.ACTIVE)

        self.env.reset()
        self.env.board[5, 2:6] = 1
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[2:6, 3] = -1
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[[2, 3, 4, 5], [0, 1, 2, 3]] = 1
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[[3, 2, 1, 0], [0, 1, 2, 3]] = -1
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.offsets[:] = 0
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.DRAW)
