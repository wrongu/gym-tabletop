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

    def test_are_players_done(self):
        self.assertEqual(self.env.are_players_done(), [False, False])

        self.env.game_status = GameStatus.DRAW
        self.assertEqual(self.env.are_players_done(), [True, True])

        self.env.game_status = GameStatus.WON
        self.assertEqual(self.env.are_players_done(), [True, True])

    def test_get_player_rewards(self):
        self.assertEqual(self.env.get_player_rewards(), [0, 0])

        self.env.game_status = GameStatus.DRAW
        self.assertEqual(self.env.get_player_rewards(), [0, 0])

        self.env.game_status = GameStatus.WON
        self.assertEqual(self.env.get_player_rewards(), [1, -1])

        self.env.current_player = 2
        self.assertEqual(self.env.get_player_rewards(), [-1, 1])
