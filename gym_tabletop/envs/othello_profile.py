from gym_tabletop.envs import GameStatus
from gym_tabletop.envs.othello import OthelloEnv
import random

def disp_act_helper(board, acts):
    for a in acts:
        board.board[a] = 3
    board.render()
    for a in acts:
        board.board[a] = 0

def self_play(verbose=False):
    board = OthelloEnv()
    i=0
    while board.game_status == GameStatus.ACTIVE:
        acts = sorted(board.get_available_actions())
        if verbose:
            disp_act_helper(board, acts)
        if len(acts) == 0:
            break
        board.step(acts[random.randint(0, len(acts)-1)])
        if verbose:
            print("BOARD HASH", i, board.hash_key())
        i+=1
    num_moves = len(board.board.nonzero()[0])-4
    return num_moves, board.hash_key()

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
