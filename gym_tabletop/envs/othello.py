from typing import List

import gym
from gym import spaces
import numpy as np
from scipy.signal import convolve2d

from gym_tabletop.envs import GameStatus


LAPLACE_FILTER = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
RAYS = np.array([[0, 1],  # east
                 [0, -1],  # west
                 [1, 0],  # south
                 [-1, 0],  # north
                 [1, 1],  # southeast
                 [1 , -1],  # southwest
                 [-1, 1],  # northeast
                 [-1, -1]  # northwest
                 ])


class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    game_symbols = [' ', '\u25cf', '\u25cb']

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[[3, 4], [3, 4]] = 2
        self.board[[3, 4], [4, 3]] = 1
        self.n_dark = 2
        self.n_light = 2
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE
        # self.action_space = spaces.Discrete(n_actions)

    def step(self, action: tuple):
        self.board[action] = self.current_player
        opp = 1 if self.current_player == 2 else 2
        hits = self._cast_rays(action)
        for hit in hits:
            pos = action + RAYS[hit]
            while self.board[tuple(pos)] == opp:
                self.board[tuple(pos)] = self.current_player
                pos += RAYS[hit]

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        self.game_status = self._evaluate_game_state()
        reward = self.get_player_rewards()
        done = self.are_players_done()
        obs = self.get_player_observations()

        return obs, reward, done, {}

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[[3, 4], [3, 4]] = 2
        self.board[[3, 4], [4, 3]] = 1
        self.n_dark = 2
        self.n_light = 2
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

    def render(self, mode='human'):
        for row in self.board:
            print([self.game_symbols[e] for e in row])

    def get_available_actions(self):
        diff = convolve2d(self.board, LAPLACE_FILTER, 'same')
        diff[self.board.nonzero()] = 0
        candidate_positions = list(zip(*diff.nonzero()))
        valid_positions = [pos for pos in candidate_positions
                           if self._is_valid_position(pos)]
        return valid_positions

    def are_players_done(self) -> List[bool]:
        if self.game_status in [GameStatus.WON, GameStatus.DRAW]:
            return [True, True]
        else:
            return [False, False]

    def get_player_rewards(self) -> List[float]:
        if self.game_status is GameStatus.WON:
            if self.n_dark > self.n_light:
                return [1, -1]
            else:
                return [-1, 1]
        else:
            return [0, 0]

    def get_player_observations(self) -> List[np.ndarray]:
        return [self.board, self.board]

    def _evaluate_game_state(self) -> GameStatus:
        self.n_dark = len(np.where(self.board == 1)[0])
        self.n_light = len(np.where(self.board == 2)[0])

        if len(self.get_available_actions()) > 0:
            return GameStatus.ACTIVE

        if self.n_dark == self.n_light:
            return GameStatus.DRAW
        else:
            return GameStatus.WON

    def _is_valid_position(self, position) -> bool:
        hits = self._cast_rays(position)
        return len(hits) > 0

    def _cast_rays(self, origin) -> List:
        opp = 1 if self.current_player == 2 else 2
        hits = []
        for i, ray in enumerate(RAYS):
            pos = origin + ray
            idx = tuple(pos)
            if np.any(pos < 0) or np.any(pos >= 8) or self.board[idx] != opp:
                continue
            pos += ray
            while np.all(pos >= 0) and np.all(pos < 8):
                piece = self.board[tuple(pos)]
                if piece == opp:
                    pos += ray
                else:
                    if piece == self.current_player:
                        hits.append(i)
                    break
        return hits
