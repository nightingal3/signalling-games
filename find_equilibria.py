import nashpy as nash
import numpy as np
from typing import Tuple, List
from itertools import combinations_with_replacement
import time
import pdb

NUM_EVENTS = 3
NUM_SIGNALS = 3

def enumerate_rewards_2x2(granularity: float = 0.1, probs: List = [0.5, 0.5], costs: List = [1, 1]) -> Tuple:
    rows = []
    increments = np.linspace(0.0, 1.0, int(1 / granularity) + 1)
    remainders = np.ones((1, int(1 / granularity) + 1)) - increments
    rows = [[i, j] for i, j in zip(increments, remainders[0])]
    
    combs = [np.array(x) for x in list(combinations_with_replacement(rows, 2))]
    reward_matrix = np.zeros((len(combs), len(combs)))
    sender_key = {}
    receiver_key = {}
    
    for i, base_sender_matrix in enumerate(combs):
        sender_matrix = (base_sender_matrix.T * probs).T
        for j, base_receiver_matrix in enumerate(combs):
            receiver_matrix = base_receiver_matrix/costs
            reward_matrix[i, j] = np.trace(np.matmul(sender_matrix, receiver_matrix))
            sender_key[(i, j)] = base_sender_matrix
            receiver_key[(i, j)] = base_receiver_matrix

    return reward_matrix, sender_key, receiver_key

if __name__ == "__main__":
    start = time.time()
    probs = [0.9, 0.1]
    costs = [1, 5]
    rewards, sender_key, receiver_key = enumerate_rewards_2x2(probs=probs, costs=costs, granularity=0.5)
    print(f"{rewards.shape} game")
    rps = nash.Game(rewards, rewards)
   
    print(rps)
    eqs = rps.support_enumeration()
    print(list(eqs))
    print(sender_key)
    print(receiver_key)
    end = time.time()
    print("time: ", end - start)

