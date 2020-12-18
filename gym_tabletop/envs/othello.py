from typing import List

import gym
from gym import spaces
import numpy as np
from scipy.signal import convolve2d

from gym_tabletop.envs import GameStatus


LAPLACE_FILTER = [[1, 1, 1],
                  [1, -8, 1],
                  [1, 1, 1]]
RAYS = [[0, 1],  # east
        [0, -1],  # west
        [1, 0],  # south
        [-1, 0],  # north
        [1, 1],  # southeast
        [1, -1],  # southwest
        [-1, 1],  # northeast
        [-1, -1]  # northwest
        ]


def is_cardinal(start, end) -> bool:
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2-x1, y2-y1
    return dx == 0 or dy == 0 or abs(dx)==abs(dy)

def is_ray_thru_point(ray_start, ray_end, pos) -> bool:
    """Helper function: returns true iff x,y coordinates of pos is colinear with (start,end) and lies between them
    """
    if pos == ray_end or pos == ray_start:
        return False
    x1, y1 = ray_start
    x2, y2 = ray_end
    xp, yp = pos
    # Colinearity test: using ray_start as the reference, check that slopes are equal. In other words,
    # (y2-y1)/(x2-x1)=(yp-y1)/(xp-x1). Avoiding divide-by-zero, this is:
    colinear = (y2 - y1) * (xp - x1) == (yp - y1) * (x2 - x1)
    if not colinear:
        return False
    # Dot product test: dot of (pos-start) x (end-start) must lie between 0 and the norm of (end-start)
    dot = (x2 - x1) * (xp - x1) + (y2 - y1) * (yp - y1)
    norm = (x2 - x1) ** 2 + (y2 - y1) ** 2
    return 0 < dot < norm

PLAYER_1 = 1
PLAYER_2 = 2

class OthelloEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    game_symbols = ['_', '\u25cf', '\u25cb', '#']

    def __init__(self):
        self.reset()
        # self.action_space = spaces.Discrete(n_actions)

    def step(self, action: tuple):
        # Add a stone to position 'action'
        self.board[action] = self.current_player

        # Find all directions to cast rays for flipping opponent stones
        if self.current_player == PLAYER_1:
            ray_endpoints = self._legal_rays_1[action]
        else:
            ray_endpoints = self._legal_rays_2[action]

        # Update candidate action positions, aka open spaces that are adjacent to some stone
        self._candidate_positions.remove(action)
        for i in range(8):
            x, y = action[0]+RAYS[i][0], action[1]+RAYS[i][1]
            if 0 <= x < 8 and 0 <= y < 8 and self.board[x, y] == 0:
                self._candidate_positions.add((x,y))

        # Remove legal rays for both players that emanate from this point. Note: this must happen after
        # _candidate_positions is updated!
        self._update_legal_cache(action, self.current_player, is_new=True)

        # Flip opponent stones along each ray. For each flipped stone, update the set of legal moves based on rays
        # emanating out from that point or passing through that point
        opp = 3-self.current_player
        for endpt in ray_endpoints:
            direction = np.sign(np.array(endpt)-action)
            pos = action + direction
            while self.board[tuple(pos)] == opp:
                self.board[tuple(pos)] = self.current_player
                self._update_legal_cache(tuple(pos), self.current_player, is_new=False)
                pos += direction

        # Switch whose turn it is
        if self.current_player == PLAYER_1:
            self.current_player = PLAYER_2
        else:
            self.current_player = PLAYER_1

        self.game_status = self._evaluate_game_state()
        reward = self.get_player_rewards()
        done = self.are_players_done()
        obs = self.get_player_observations()

        return obs, reward, done, {}

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.board[[3, 4], [3, 4]] = PLAYER_2
        self.board[[3, 4], [4, 3]] = PLAYER_1
        self._compute_edge_set()
        self.n_dark = 2
        self.n_light = 2
        self.current_player = PLAYER_1
        self.game_status = GameStatus.ACTIVE
        self._rebuild_legal_cache()

    def render(self, mode='human'):
        print("======="+self.game_symbols[self.current_player]*2+"=======")
        for row in self.board:
            print(" ".join(self.game_symbols[e] for e in row))
        print("================")

    def _compute_edge_set(self):
        diff = convolve2d(self.board, LAPLACE_FILTER, 'same')
        diff[self.board.nonzero()] = 0
        self._candidate_positions = set(zip(*diff.nonzero()))

    def get_available_actions(self):
        if self.current_player == PLAYER_1:
            return list(k for k in self._legal_rays_1.keys() if len(self._legal_rays_1[k]) > 0)
        else:
            return list(k for k in self._legal_rays_2.keys() if len(self._legal_rays_2[k]) > 0)

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
        self.n_dark = len(np.where(self.board == PLAYER_1)[0])
        self.n_light = len(np.where(self.board == PLAYER_2)[0])

        if len(self.get_available_actions()) > 0:
            return GameStatus.ACTIVE

        if self.n_dark == self.n_light:
            return GameStatus.DRAW
        else:
            return GameStatus.WON

    def _cast_rays(self, origin, capture_color, terminal_color, offset=1) -> List:
        hits = []
        for i, ray in enumerate(RAYS):
            x,y = origin[0]+offset*ray[0], origin[1]+offset*ray[1]
            if not (0 <= x < 8 and 0 <= y < 8) or self.board[x,y] != capture_color:
                continue
            x, y = x+ray[0], y+ray[1]
            while 0 <= x < 8 and 0 <= y < 8:
                piece = self.board[x,y]
                if piece == capture_color:
                    x, y = x+ray[0], y+ray[1]
                else:
                    if piece == terminal_color:
                        hits.append((x,y))
                    break
        return hits

    def hash_key(self):
        coord_1 = np.where(self.board.flatten() == 1)[0]
        coord_2 = np.where(self.board.flatten() == 2)[0]
        return hash(tuple(coord_1) + tuple(coord_2))

    def _rebuild_legal_cache(self):
        """The 'legal cache' refers to two dictionaries, _legal_rays_2 and _legal_rays_1. The keys of these
        dictionaries are open positions on the board. So if (3,4) is a key in _legal_rays_2, then we know (3,4) is
        an empty space and a legal move for player 2. The values of the dicts are a set of ray endpoints. For instance,
        if _legal_rays_2[(3,4)] = {(3,6), (5,6)} then this means that there are two 'rays' connecting (3,4) to both
        (3,6) (horizontal) and (5,6) (diagonal) where all stones along this ray will be captured. This means that (3,4)
        is empty, (3,6) is a white stone (player 2), and everything along the ray between them is dark (player 1).

        This function is relatively slow - it rebuilds the set of legal move rays from scratch. For fast state updates,
        see _update_legal_cache.
        """
        self._legal_rays_1 = dict()
        self._legal_rays_2 = dict()
        for (x,y) in self._candidate_positions:
            assert self.board[x,y] == 0
            # Cast rays from all empty edge positions, separately per player
            endpoints_1 = self._cast_rays((x,y), capture_color=PLAYER_2, terminal_color=PLAYER_1)
            if (x,y) in self._legal_rays_1:
                self._legal_rays_1[(x,y)].update(endpoints_1)
            else:
                self._legal_rays_1[(x,y)] = set(endpoints_1)

            endpoints_2 = self._cast_rays((x,y), capture_color=PLAYER_1, terminal_color=PLAYER_2)
            if (x,y) in self._legal_rays_2:
                self._legal_rays_2[(x,y)].update(endpoints_2)
            else:
                self._legal_rays_2[(x,y)] = set(endpoints_2)

    def _update_legal_cache(self, location:tuple, my_color:int, is_new:bool):
        """Update the cache of legal moves' rays (_legal_rays_2 and _legal_rays_1) given that a stone was just
        placed or flipped at 'location' now owned by 'player'. Set is_new=True if this space was previously empty. Set
        is_new=False if it's a previously occupied location that is being flipped
        """
        my_legal_rays = self._legal_rays_1 if my_color == PLAYER_1 else self._legal_rays_2
        their_legal_rays = self._legal_rays_2 if my_color == PLAYER_1 else self._legal_rays_1

        # Need to check the following cases. (1) and (2) are checked no matte rwhat. (3) and (4) only need checking if
        # this stone is being 'flipped' from an existing opponent stone.
        # 1. This location may now be the endpoint for newly legal moves by the current player, which need adding.
        # 2. This location may now be capturable by the opponent by connecting a ray between them and a blank space.
        # 3. This location may have been the endpoint of a ray or rays owned by the opponent, which need removing.
        # 4. This location may have previously been capturable by the player playing elsewhere.

        # Handle case 1: This location may have just become a new ray endpoint for the current player. Do a reverse
        # ray cast, starting from this position and searching for a path through opponent stones to an open space
        # (a '0' at the terminal point of the ray)
        open_endpoints = self._cast_rays(location, capture_color=3-my_color, terminal_color=0)
        for e in open_endpoints:
            if e in my_legal_rays:
                my_legal_rays[e].add(location)
            else:
                my_legal_rays[e] = {location}

        # Handle case 2: this location may be newly capturable by the opponent. Search for rays out from here that
        # terminate on an opponent stone. Then, search in the reverse direction for an open endpoint.
        # Note zero 'offset' here since the opponent might be adjacent to this stone...
        cardinal_opponents = self._cast_rays(location, capture_color=my_color, terminal_color=3-my_color, offset=0)
        for opp in cardinal_opponents:
            for empty_pos in self._candidate_positions:
                if is_cardinal(empty_pos, opp) and is_ray_thru_point(empty_pos, opp, location):
                    # Final (slow) test: are all stones between 'location' and 'empty_pos' also 'my_color'?
                    dx, dy = np.sign(empty_pos[0]-opp[0]), np.sign(empty_pos[1]-opp[1])
                    x, y = location
                    while (x,y) != empty_pos:
                        if self.board[x,y] != my_color:
                            break
                        x, y = x+dx, y+dy
                    else:
                        # 'else' after 'while' should be read as a 'nobreak' clause, here meaning that the ray did
                        # indeed contain all 'my_color' stones. Add a ray for the opponent from empty_pos to opp,
                        # passing through 'location'
                        if empty_pos in their_legal_rays:
                            their_legal_rays[empty_pos].add(opp)
                        else:
                            their_legal_rays[empty_pos] = {opp}

        # Handle things that only need updating depending on if this is the new endpoint stone or not
        if not is_new:
            # Handle case 3: remove all opponent rays that had ended on this location.
            for start in their_legal_rays.keys():
                if location in their_legal_rays[start]:
                    their_legal_rays[start].remove(location)

            # Handle case 4: Flipping this location invalidated all other of my rays that had previously gone through it
            for start, endpts in my_legal_rays.items():
                if is_cardinal(start, location):
                    end_to_remove = {end for end in endpts if is_ray_thru_point(start, end, location)}
                    my_legal_rays[start] -= end_to_remove
        else:
            # This is a new stone. This location must be removed as a key from all legal move dicts -- it is now
            # occupied! It cannot have been in the set of any endpoint values since it was empty.
            if location in my_legal_rays:
                del my_legal_rays[location]
            if location in their_legal_rays:
                del their_legal_rays[location]
