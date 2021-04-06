import csv
import nashpy as nash
import numpy as np
from fractions import Fraction
from typing import Tuple, List
from itertools import combinations_with_replacement
from collections import Counter
import time
import pdb
from pprint import pprint

NUM_EVENTS = 3
NUM_SIGNALS = 3

def enumerate_rewards_2x2(granularity: float = 0.1, probs: List = [0.5, 0.5], costs: List = [1, 1]) -> Tuple:
    rows = []
    increments = np.linspace(0.0, 1.0, int(1 / granularity) + 1)
    remainders = np.ones((1, int(1 / granularity) + 1)) - increments
    rows = [[i, j] for i, j in zip(increments, remainders[0])]
    
    combs_base = [np.array(x) for x in list(combinations_with_replacement(rows, 2))]
    combs = []
    seen = []
    for c in combs_base:
        if c.tolist() not in seen:
            combs.append(c)
        seen.append(c.tolist())
        seen.append(np.flip(c, axis=1).tolist())

    reward_matrix = np.zeros((len(combs), len(combs)))
    sender_key = {}
    receiver_key = {}
    
    for i, base_sender_matrix in enumerate(combs):
        sender_matrix = (base_sender_matrix.T * probs).T
        for j, base_receiver_matrix in enumerate(combs):
            receiver_matrix = base_receiver_matrix/costs
            reward_matrix[i, j] = np.trace(np.matmul(sender_matrix, receiver_matrix))
            sender_key[i] = base_sender_matrix
            receiver_key[j] = base_receiver_matrix

    return reward_matrix, sender_key, receiver_key

def enumerate_rewards_nxm(n: int, m: int, probs: List, costs: List, granularity: float = 0.1) -> Tuple:
    rows = generate_allocations(m, granularity=granularity)
    cols = generate_allocations(n, granularity=granularity)

    combs_sender = [np.array(x) for x in list(combinations_with_replacement(rows, n))]
    combs_receiver = [np.array(x) for x in list(combinations_with_replacement(rows, m))]

    reward_matrix = np.zeros((len(combs_sender), len(combs_receiver)))
    sender_key = {}
    receiver_key = {}

    for i, base_sender_matrix in enumerate(combs_sender):
        sender_matrix = (base_sender_matrix.T * probs).T
        for j, base_receiver_matrix in enumerate(combs_receiver):
            receiver_matrix = (base_receiver_matrix.T / costs).T
            reward_matrix[i, j] = np.trace(np.matmul(sender_matrix, receiver_matrix))
            sender_key[i] = base_sender_matrix
            receiver_key[j] = base_receiver_matrix

    return reward_matrix, sender_key, receiver_key

# generate all allocations between m signals (with a mass of 1), with specified increment
# based on this answer: https://stackoverflow.com/questions/27586404/how-to-efficiently-get-all-combinations-where-the-sum-is-10-or-below-in-python
def generate_allocations(m: int, granularity: float = 0.1) -> List:
    def f(r, n, t, acc=[]):
        if t == 0:
            if n >= 0:
                yield acc
            return
        for x in r:
            if x > n: 
                break
            for lst in f(r, n-x, t-1, acc + [x]):
                yield lst

    n = int(1 / granularity)
    print("N: ", n)
    allocations = []
    for xs in f(range(n+1), n, m):
        if sum([i * granularity for i in xs]) == 1:
            print([i * granularity for i in xs])
            allocations.append([i * granularity for i in xs])

    return allocations


def generate_lrs_game_file(reward_matrix: np.ndarray, trial_name: str, convert_to_frac: bool = True) -> None:
    with open(f"{trial_name}.txt", "w") as out_f:
        writer = csv.writer(out_f, delimiter=" ")
        out_f.write(f"{reward_matrix.shape[0]} {reward_matrix.shape[1]}\n\n")
        for row in reward_matrix:
            frac_row = [str(Fraction(item)) for item in row]
            writer.writerow(frac_row)

        out_f.write("\n\n")
        for row in reward_matrix:
            frac_row = [str(Fraction(item)) for item in row]
            writer.writerow(frac_row)
        

if __name__ == "__main__":
    start = time.time()
    probs = [0.7, 0.2, 0.05, 0.05]
    costs = [1, 5]

    rewards, sender_key, receiver_key = enumerate_rewards_nxm(n=4, m=2, probs=probs, costs=costs, granularity=0.25)
    #rewards, sender_key, receiver_key = enumerate_rewards_2x2(probs=probs, costs=costs, granularity=0.2)
    print(rewards)
    generate_lrs_game_file(rewards, "trial_4", convert_to_frac=True)
    assert False
    rps = nash.Game(rewards, rewards)
   
    print(rps)
    eqs = rps.support_enumeration()
    print(list(eqs))
    print(sender_key)
    print(receiver_key)
    end = time.time()
    print("time: ", end - start)

