import csv
import numpy as np
from fractions import Fraction
from typing import Tuple, List
from itertools import combinations_with_replacement
from collections import Counter
import pdb
import argparse
import pickle

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

    combs_sender_base = [np.array(x) for x in list(combinations_with_replacement(rows, n))]
    combs_receiver_base = [np.array(x) for x in list(combinations_with_replacement(cols, m))]

    seen_sender = []
    combs_sender = []
    for c in combs_sender_base:
        if c.tolist() not in seen_sender:
            combs_sender.append(c)
        seen_sender.append(c.tolist())
        seen_sender.append(np.flip(c, axis=1).tolist())

    seen_receiver = []
    combs_receiver = []
    for c in combs_receiver_base:
        if c.tolist() not in seen_receiver:
            combs_receiver.append(c)
        seen_receiver.append(c.tolist())
        seen_receiver.append(np.flip(c, axis=1).tolist())

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
    allocations = []
    for xs in f(range(n+1), n, m):
        if sum([i * granularity for i in xs]) == 1:
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
        
def main(N: int, M: int, granularity: float, probs: List, costs: List, out_filename: str, key_filename: str) -> None:
    rewards, sender_key, receiver_key = enumerate_rewards_nxm(n=N, m=M, probs=probs, costs=costs, granularity=granularity)

    generate_lrs_game_file(rewards, out_filename, convert_to_frac=True)

    with open(key_filename, "wb") as key_file:
        master_key = {"sender": sender_key, "receiver": receiver_key}
        pickle.dump(master_key, key_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the reward matrix for a game with varying costs, probabilities, states, and signals. Writes to a file for processing.")
    parser.add_argument("-n", "--states", type=int, help="The number of states in the game", required=True)
    parser.add_argument("-m", "--signals", type=int, help="The number of signals in the game", required=True)
    parser.add_argument("-g", type=float, help="Granularity of allocations between states/signals", default=0.25)
    parser.add_argument("-i", type=int, help="Number of increments allowed for functions of sender and receiver. Alternative to granularity.")
    parser.add_argument("-p", "--probs", type=float, nargs="+", help="Probabilities of states (must match number of states)")
    parser.add_argument("-c", "--costs", type=float, nargs="+", help="Costs of signals (must match number of signals)")
    parser.add_argument("-o", type=str, help="Name of file to write game file to (should be a txt file)", required=True)
    parser.add_argument("-k", "--key-out", type=str, help="Name of file to write matrix key to")

    args = parser.parse_args()
    N = args.states
    M = args.signals
    granularity = args.g
    if args.i:
        granularity = 1/args.i
    probs = args.probs
    costs = args.costs
    out_filename = f"nash_trials/{args.o}"
    key_filename = args.key_out

    if probs is None:
        probs = [1/args.n for i in range(args.n)]
    if sum(probs) != 1:
        probs = [i/sum(probs) for i in probs]
    if costs is None:
        costs = [1 for i in range(args.m)]
    if key_filename is None:
        key_filename = f"{out_filename}_key.p"

    main(N, M, granularity, probs, costs, out_filename, key_filename)
   
        
   
