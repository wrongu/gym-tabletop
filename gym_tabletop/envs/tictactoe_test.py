import unittest

from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.tictactoe import TicTacToeEnv


class TestTicTacToe(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TicTacToeEnv()

    def test_get_possible_actions(self):
        expected = [(0, 0), (0, 1), (0, 2),
                    (1, 0), (1, 1), (1, 2),
                    (2, 0), (2, 1), (2, 2)]
        self.assertEqual(self.env.get_available_actions(), expected)

        self.env.board[:] = 1
        self.assertEqual(self.env.get_available_actions(), [])

    def test_evaluate_state(self):
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.ACTIVE)

        self.env.board[:] = [[1, 1, 1], [0, 0, 0], [0, 0, 0]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.board[:] = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.board[:] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.board[:] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.board[:] = [[1, -1, -1], [-1, 1, 1], [-1, 1, -1]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.DRAW)

        self.env.board[:] = [[1, -1, -1], [0, 1, 1], [-1, 1, -1]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.ACTIVE)

    def test_is_terminal(self):
        self.assertFalse(self.env.is_terminal())

        self.env.game_status = GameStatus.DRAW
        self.assertTrue(self.env.is_terminal())

        self.env.game_status = GameStatus.WON
        self.assertTrue(self.env.is_terminal())

    def test_get_reward(self):
        self.assertEqual(self.env.get_reward(), 0)

        self.env.game_status = GameStatus.DRAW
        self.assertEqual(self.env.get_reward(), 0)

        self.env.game_status = GameStatus.WON
        self.assertEqual(self.env.get_reward(), 1)
