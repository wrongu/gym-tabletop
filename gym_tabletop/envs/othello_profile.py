from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.othello import OthelloEnv
import random
import numpy as np

def self_play():
    board = OthelloEnv()
    while board.game_status == GameStatus.ACTIVE:
        acts = board.get_available_actions()
        if len(acts) == 0:
            break
        board.step(acts[random.randint(0, len(acts)-1)])
    coord_1 = np.where(board.board.flatten() == 1)[0]
    coord_2 = np.where(board.board.flatten() == 2)[0]
    board_hash = hash(tuple(coord_1)) ^ hash(tuple(coord_2))
    return len(coord_1)+len(coord_2)-4, board_hash

if __name__ == '__main__':
    import time
    tstart = time.time()
    ngames = 10
    hashes, lengths = [0]*ngames, [0]*ngames
    for n in range(ngames):
        random.seed(102938475+n)
        lengths[n], hashes[n] = self_play()
    elapsed = time.time()-tstart

    print(f"Played {ngames} games ({sum(lengths)} total turns) in {elapsed} seconds")
    print(f"{elapsed/sum(lengths)} seconds per move")
    print(hashes)
