from typing import List

import gym
from gym import spaces
import numpy as np
from scipy.signal import convolve2d

from gym_tabletop.envs import GameStatus


class ConnectFourEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    game_symbols = [' ', '\u25cf', '\u25cb']

    def __init__(self):
        self.board = np.zeros((6, 7), dtype=np.int)
        self.offsets = 5*np.ones(7, dtype=np.int)
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(0, 2, shape=(6, 7), dtype=np.int)

    def step(self, action):
        values = [0, 1, -1]
        self.board[self.offsets[action], action] = values[self.current_player]
        self.offsets[action] -= 1
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
        self.board = np.zeros((6, 7), dtype=np.int)
        self.offsets = 5*np.ones(7, dtype=np.int)
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

    def render(self, mode='human'):
        for row in self.board:
            print([self.game_symbols[e] for e in row])

    def get_available_actions(self) -> list:
        actions = list(self.offsets.nonzero()[0])
        return actions

    def _evaluate_game_state(self) -> GameStatus:
        checks = np.vstack((
            convolve2d(self.board, np.ones((1, 4)), 'same'),
            convolve2d(self.board, np.ones((4, 1)), 'same'),
            convolve2d(self.board, np.eye(4), 'same'),
            convolve2d(self.board, np.fliplr(np.eye(4)), 'same'),
        ))

        if np.any(np.abs(checks) == 4):
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
