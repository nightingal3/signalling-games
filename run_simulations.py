import numpy as np
from scipy.stats import entropy
from typing import List
import pdb
import subprocess

from price_of_anarchy import *
from enumerate_rewards import main

def gen_random_probs(n: int) -> List:
    unnormalized = np.random.randint(low=1, high=100, size=n)
    return unnormalized/sum(unnormalized)
    
def gen_random_costs(m: int) -> List:
    return np.random.randint(low=1, high=100, size=m)

def random_simulation(n: int, m: int, granularity: float, out_filename: str, key_filename: str, num_simulations: int = 1000) -> None:
    x_coords = []
    y_coords = []
    prices_of_anarchy = []
    prices_of_stability = []

    for sim_num in range(num_simulations):
        probs = gen_random_probs(n)
        costs = gen_random_costs(m)

        entropy_probs = entropy(probs)
        entropy_costs = entropy(costs/sum(costs))
        
        # create the input files to send to lrsnash
        main(n, m, granularity, probs, costs, out_filename, key_filename)

        # solve for nash equilibria - ensure lrsnash is installed 
        solve_ne = subprocess.run(["lrsnash", out_filename, ">", f"{out_filename[:-4]}_sol.txt"], check=True)

        # get PoA/PoS
        pure_strategies = read_nash_file(f"{out_filename[:-4]}_sol.txt", key_filename, pure_only=True)
        PoA = price_of_anarchy(pure_strategies, n, m)
        PoS = price_of_stability(pure_strategies, n, m)

        x_coords.append(entropy_probs)
        y_coords.append(entropy_costs)
        prices_of_anarchy.append(PoA)
        prices_of_stability.append(PoS)

    return x_coords, y_coords, prices_of_anarchy, prices_of_stability




if __name__ == "__main__":
    random_simulation(3, 2, "test.txt")

    