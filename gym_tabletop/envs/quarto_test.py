import unittest

import numpy as np

from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.quarto import QuartoEnv


class TestQuarto(unittest.TestCase):
    def setUp(self) -> None:
        self.env = QuartoEnv()

    def test_get_possible_actions_pick_phase(self):
        expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.assertEqual(self.env.get_available_actions(), expected)

        self.env.pieces = []
        self.assertEqual(self.env.get_available_actions(), [])

    def test_get_possible_actions_place_phase(self):
        self.env.active_piece = 0

        expected = [(0, 0), (0, 1), (0, 2), (0, 3),
                    (1, 0), (1, 1), (1, 2), (1, 3),
                    (2, 0), (2, 1), (2, 2), (2, 3),
                    (3, 0), (3, 1), (3, 2), (3, 3)]
        self.assertEqual(self.env.get_available_actions(), expected)

        self.env.board[:] = 0
        self.assertEqual(self.env.get_available_actions(), [])

    def test_evaluate_game_state(self):
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.ACTIVE)

        self.env.board[0, :] = [0, 1, 2, 4]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[:, 2] = [15, 7, 3, 1]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[[0, 1, 2, 3], [0, 1, 2, 3]] = [8, 9, 10, 11]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[[3, 2, 1, 0], [0, 1, 2, 3]] = [4, 5, 12, 13]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.WON)

        self.env.reset()
        self.env.board[:] = [[15, 0, 14, 1],
                             [4, 11, 5, 10],
                             [13, 12, 8, 7],
                             [3, 9, 6, 2]]
        self.assertEqual(self.env._evaluate_game_state(), GameStatus.DRAW)

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