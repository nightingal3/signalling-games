import csv
import numpy as np
import pickle
from fractions import Fraction
import pdb
from typing import List
from pprint import pprint

def fix_spacing(filename: str) -> None:
    fin = open(filename, "r")
    fout = open(f"{filename[:-4]}_fixed.txt", "w")

    for line in fin:
        fout.write(' '.join(line.split()))
        fout.write("\n")
        
    fin.close()
    fout.close()

# puts the output generated by lrsnash into a more readable/understandable format
def read_nash_file(filename: str, key_filename: str, pure_only: bool = True) -> None:
    key = pickle.load(open(key_filename, "rb"))
    strategies = [] # format as: {sender matrix, receiver matrix, payoff}
    pure_strategies = []
    num_nash_equilibria = 0
    num_pure_nash_equilibria = 0
    with open(filename, "r") as trial_file:
        reader = csv.reader(trial_file, delimiter=" ")
        receiver_matrices = []
        for line in reader:
            if len(line) == 0 or line[0] == "*lrsnash:lrslib":
                continue
            line = [i for i in line if i != ""]
            if line[0] == "2":
                strategy_vec = [float(Fraction(s)) for s in line[1:-1]]
                if pure_only:
                    if any([True if i != 0 and i != 1 else False for i in strategy_vec]):
                        continue
                    rec_matrix = key["receiver"][strategy_vec.index(1)]
                    receiver_matrices.append(rec_matrix)
                else:
                    raise NotImplementedError
            elif line[0] == "1":
                strategy_vec = [float(Fraction(s)) for s in line[1:-1]]
                payoff = float(Fraction(line[-1]))
                if pure_only:
                    if any([True if i != 0 and i != 1 else False for i in strategy_vec]):
                        receiver_matrices = []
                        continue
                    sender_matrix = key["sender"][strategy_vec.index(1)]
                    for receiver_matrix in receiver_matrices:
                        pure_strategies.append({
                            "sender_matrix": sender_matrix,
                            "receiver_matrix": receiver_matrix,
                            "payoff": payoff
                        })
            else:
                continue

    if pure_only:
        return pure_strategies

def price_of_anarchy(pure_strategies: List, n: int, m: int, probs: List, costs: List) -> float:
    # do general case later...
    if n == 2 and m == 2: 
        base_sender_matrix = np.identity(2)
        base_sender_matrix_alt = np.flip(base_sender_matrix, axis=1)
        base_receiver_matrix = np.identity(2)
        base_receiver_matrix_alt = np.flip(base_receiver_matrix, axis=1)
        sender_matrix = (base_sender_matrix.T * probs).T
        receiver_matrix = base_receiver_matrix/costs
        alt_sender_matrix = (base_sender_matrix_alt.T * probs).T
        alt_receiver_matrix = (base_receiver_matrix_alt.T * probs).T
        max_payoff = max(np.trace(np.matmul(sender_matrix, receiver_matrix)), np.trace(np.matmul(alt_sender_matrix, alt_receiver_matrix)))

    elif n == 3 and m == 2:
        signals_by_cost = sorted(costs)
        highest_probs = sorted(probs, reverse=True)
    
        best_events = highest_probs[:2]
        best_indices_sender = np.where(np.in1d(probs, best_events))[0]
        cheapest_signal, other = list(costs).index(signals_by_cost[0]), list(costs).index(signals_by_cost[1])
        best_indices_receiver = [cheapest_signal, other]
        nonzero_indices_s = (tuple(best_indices_sender), tuple(best_indices_receiver))
        nonzero_indices_r = (tuple(best_indices_receiver), tuple(best_indices_sender))

        base_sender_matrix = np.zeros((n, m))
        base_sender_matrix[nonzero_indices_s] = 1

        base_receiver_matrix = np.zeros((m, n))
        base_receiver_matrix[nonzero_indices_r] = 1

        sender_matrix = (base_sender_matrix.T * probs).T
        receiver_matrix = ((base_receiver_matrix.T)/costs).T

        max_payoff = np.trace(np.matmul(sender_matrix, receiver_matrix))

    worst_payoff = float("inf")
    worst_sender = None
    worst_receiver = None
    
    for strategy in pure_strategies:
        if strategy["payoff"] < worst_payoff:
            worst_sender = strategy["sender_matrix"]
            worst_receiver = strategy["receiver_matrix"] 
            worst_payoff = strategy["payoff"]

    return worst_payoff/max_payoff

def price_of_stability(pure_strategies: List, n: int, m: int, probs: List, costs: List) -> float:
    if n == 2 and m == 2:
        base_sender_matrix = np.identity(2)
        base_sender_matrix_alt = np.flip(base_sender_matrix, axis=1)
        base_receiver_matrix = np.identity(2)
        base_receiver_matrix_alt = np.flip(base_receiver_matrix, axis=1)
        sender_matrix = (base_sender_matrix.T * probs).T
        receiver_matrix = base_receiver_matrix/costs
        alt_sender_matrix = (base_sender_matrix_alt.T * probs).T
        alt_receiver_matrix = (base_receiver_matrix_alt.T * probs).T
        max_payoff = max(np.trace(np.matmul(sender_matrix, receiver_matrix)), np.trace(np.matmul(alt_sender_matrix, alt_receiver_matrix)))
    
    if n == 3 and m == 2:
        signals_by_cost = sorted(costs)
        highest_probs = sorted(probs, reverse=True)
    
        best_events = highest_probs[:2]
        best_indices_sender = np.where(np.in1d(probs, best_events))[0]
        cheapest_signal, other = list(costs).index(signals_by_cost[0]), list(costs).index(signals_by_cost[1])
        best_indices_receiver = [cheapest_signal, other]
        nonzero_indices_s = (tuple(best_indices_sender), tuple(best_indices_receiver))
        nonzero_indices_r = (tuple(best_indices_receiver), tuple(best_indices_sender))

        base_sender_matrix = np.zeros((n, m))
        base_sender_matrix[nonzero_indices_s] = 1

        base_receiver_matrix = np.zeros((m, n))
        base_receiver_matrix[nonzero_indices_r] = 1

        sender_matrix = (base_sender_matrix.T * probs).T
        receiver_matrix = ((base_receiver_matrix.T)/costs).T

        max_payoff = np.trace(np.matmul(sender_matrix, receiver_matrix))


    best_payoff = -float("inf")
    best_sender = None
    best_receiver = None

    for strategy in pure_strategies:
        if strategy["payoff"] > best_payoff:
            best_sender = strategy["sender_matrix"]
            best_receiver = strategy["receiver_matrix"] 
            best_payoff = strategy["payoff"]

    return best_payoff/max_payoff



if __name__ == "__main__":
    fix_spacing("trial_2_results.txt")
    pure_strats = read_nash_file("trial_2_results_fixed.txt", "nash_trials/testing_key.p", "out.txt")
    price_of_anarchy(pure_strats)
        