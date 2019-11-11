from typing import List

import gym
from gym import spaces
import numpy as np

from gym_tabletop.envs import GameStatus


class QuartoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.board = np.nan*np.zeros((4, 4))
        self.pieces = list(range(16))
        self.active_piece = None
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE
        # self.action_space = spaces.Discrete(n_actions)

    def step(self, action):
        if self.active_piece is None:
            self.active_piece = self.pieces.pop(self.pieces.index(action))
        else:
            self.board[action] = self.active_piece
            self.active_piece = None
            self.game_status = self._evaluate_game_state()

        reward = self.get_player_rewards()
        done = self.are_players_done()
        obs = self.get_player_observations()

        if self.active_piece is not None:
            if self.current_player == 1:
                self.current_player = 2
            else:
                self.current_player = 1

        return obs, reward, done, {}

    def reset(self):
        self.board = np.nan*np.zeros((4, 4))
        self.pieces = list(range(16))
        self.active_piece = None
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

    def render(self, mode='human'):
        for row in self.board:
            print(['----' if np.isnan(e) else np.binary_repr(int(e), 4)
                   for e in row])

    def get_available_actions(self):
        if self.active_piece is None:
            return self.pieces
        else:
            return list(zip(*np.where(np.isnan(self.board))))

    def are_players_done(self) -> List[bool]:
        if self.game_status in [GameStatus.WON, GameStatus.DRAW]:
            return [True, True]
        else:
            return [False, False]

    def get_player_rewards(self) -> List[float]:
        if self.game_status is GameStatus.WON:
            if self.current_player == 1:
                return [1, -1]
            else:
                return [-1, 1]
        else:
            return [0, 0]

    def get_player_observations(self) -> List[np.ndarray]:
        return [self.board, self.board]

    def _evaluate_game_state(self) -> GameStatus:
        lines = np.vstack((
            self.board,  # rows
            self.board.T,  # columns
            np.diagonal(self.board),  # \-diagonal
            np.diagonal(np.fliplr(self.board))  # /-diagonal
        ))
        for line in lines:
            if self._common_bit(line):
                return GameStatus.WON

        if len(self.get_available_actions()) == 0:
            return GameStatus.DRAW

        return GameStatus.ACTIVE

    @staticmethod
    def _common_bit(x) -> bool:
        if np.any(np.isnan(x)):
            return False
        x = x.astype(np.uint8)
        for i in range(4):
            bit = np.bitwise_and(x, 2**i)
            if np.all(bit == 0) or np.all(bit == 2**i):
                return True
        return False
