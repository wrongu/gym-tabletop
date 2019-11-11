from typing import List

import gym
from gym import spaces
import numpy as np

from gym_tabletop.envs import GameStatus


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Discrete(3)))
        self.observation_space = spaces.Box(-1, 1, shape=(3, 3), dtype=np.int)

    def step(self, action):
        game_symbols = [0, 1, -1]
        self.board[action] = game_symbols[self.current_player]
        self.game_status = self._evaluate_game_state()
        done = self.are_players_done()
        reward = self.get_player_rewards()
        obs = self.get_player_observations()

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        return obs, reward, done, {}

    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.int)
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

    def render(self, mode='human'):
        symbols = [' ', 'X', 'O']
        for row in self.board:
            print([symbols[e] for e in row])

    def get_available_actions(self) -> list:
        actions = list(zip(*np.where(self.board == 0)))
        return actions

    def _evaluate_game_state(self) -> GameStatus:
        lines = np.r_[
            np.sum(self.board, axis=0),  # columns
            np.sum(self.board, axis=1),  # rows
            np.sum(np.diagonal(self.board)),  # \-diagonal
            np.sum(np.diagonal(np.fliplr(self.board)))  # /-diagonal
        ]

        if np.any(np.abs(lines) == 3):
            return GameStatus.WON
        elif len(self.get_available_actions()) == 0:
            return GameStatus.DRAW
        else:
            return GameStatus.ACTIVE

    def are_players_done(self) -> List[bool]:
        done = self.game_status in [GameStatus.WON, GameStatus.DRAW]
        return [done, done]

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
