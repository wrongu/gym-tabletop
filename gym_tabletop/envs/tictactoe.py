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
        self.board[action] = self.current_player
        self.game_status = self._evaluate_game_state()
        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

        done = self.is_terminal()
        reward = self.get_reward()

        return self.board, reward, done, {}

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.game_status = GameStatus.ACTIVE

    def render(self, mode='human'):
        print(self.board)

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

    def is_terminal(self) -> bool:
        return self.game_status in [GameStatus.WON, GameStatus.DRAW]

    def get_reward(self) -> float:
        if self.game_status is GameStatus.WON:
            return 1
        else:
            return 0
