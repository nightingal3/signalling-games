import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from typing import List
import pdb
import subprocess
import time

from price_of_anarchy import *
from enumerate_rewards import main
from timeout import *

TRIALS_DIR = "nash_trials"
FIGURES_DIR = "nash_figures"

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
    all_probs = []
    all_costs = []

    # randomly generated points
    for sim_num in range(num_simulations):
        start = time.time()
        probs = gen_random_probs(n)
        costs = gen_random_costs(m)

        entropy_probs = entropy(probs)
        entropy_costs = entropy(costs/sum(costs))
        
        out_fname = f"{out_filename[:-4]}_{sim_num}"
        key_fname = f"{key_filename[:-2]}_{sim_num}.p"
        print("probs: ", probs)
        print("costs: ", costs)

        try:
            PoA, PoS = sim_with_params(n, m, granularity, probs, costs, out_fname, key_fname)
        except:
            continue

        x_coords.append(entropy_probs)
        y_coords.append(entropy_costs)
        prices_of_anarchy.append(PoA)
        prices_of_stability.append(PoS)
        all_probs.append(probs)
        all_costs.append(costs)
        end = time.time()
        print("time: ", end - start)
        
        print(PoA)
        print(PoS)

    # add a few anchor points for the extremes
    simulations = [
        {
            "probs": [1/n  for _ in range(n)],
            "costs": [1 for _ in range(m)],
            "name": "trial_ext_1"
        },
        {
            "probs": [0.99] + [0.01/(n - 1) for _ in range(n - 1)],
            "costs": [1 for _ in range(m)],
            "name": "trial_ext_2"
        },
        {
            "probs": [1/n for _ in range(n)],
            "costs": [10 ** (i + 1) for i in range(m)],
            "name": "trial_ext_3"
        },
        {
            "probs": [0.99] + [0.01/(n - 1) for _ in range(n - 1)],
            "costs": [10 ** (i + 1) for i in range(m)],
            "name": "trial_ext_4"
        },
        {
            "probs": [1] + [0] * (n - 1),
            "costs": [1 for _ in range(m)],
            "name": "trial_ext_5"
        },
        {
            "probs": [1] + [0] * (n - 1),
            "costs": [10 ** (i + 1) for i in range(m)],
            "name": "trial_ext_6"
        }
    ]

    for sim in simulations:
        name = sim["name"]
        try:
            PoA, PoS = sim_with_params(n, m, granularity, sim["probs"], sim["costs"], f"{out_filename[:-4]}_{name}", f"{key_filename[:-2]}_{name}")
        except:
            continue
        x_coords.append(entropy(sim["probs"]))
        y_coords.append(entropy(sim["costs"]))
        prices_of_anarchy.append(PoA)
        prices_of_stability.append(PoS)
        all_probs.append(sim["probs"])
        all_costs.append(sim["costs"])

    return x_coords, y_coords, prices_of_anarchy, prices_of_stability, all_probs, all_costs

@timeout(360)
def sim_with_params(n: int, m: int, granularity: float, probs: List, costs: List, out_fname: str, key_fname: str) -> tuple:
    # create the input files to send to lrsnash
    main(n, m, granularity, probs, costs, f"{TRIALS_DIR}/{out_fname}", f"{TRIALS_DIR}/{key_fname}")

    # solve for nash equilibria - ensure lrsnash is installed 
    try:
        with open(f"{TRIALS_DIR}/{out_fname}_sol.txt", "w") as sol_f:
            solve_ne = subprocess.run(["lrsnash", f"{TRIALS_DIR}/{out_fname}.txt"], check=True, stdout=sol_f)
    except:
        raise TimeoutError

    # get PoA/PoS
    pure_strategies = read_nash_file(f"{TRIALS_DIR}/{out_fname}_sol.txt", f"{TRIALS_DIR}/{key_fname}", pure_only=True)
    PoA = price_of_anarchy(pure_strategies, n, m, probs, costs)
    PoS = price_of_stability(pure_strategies, n, m, probs, costs)

    return PoA, PoS

def graph_entropy_and_ne(x_coords: List, y_coords: List, z_coords: List, title: str = "Price of Anarchy", out_filename: str = "test.png") -> None:
    norm = SqueezedNorm(vmin=0, vmax=max(z_coords), mid=1)
    plt.tricontourf(x_coords, y_coords, z_coords, cmap="viridis", norm=norm, levels=[0, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 8, 16])
    plt.xlabel("Entropy of probability distribution")
    plt.ylabel("Entropy of signal costs")
    plt.colorbar()
    plt.title(title)

    plt.savefig(f"{FIGURES_DIR}/{out_filename}")


# credit to ImportanceOfBeingEarnest for this graphing class: 
# https://stackoverflow.com/questions/44432693/change-colorbar-gradient-in-matplotlib
class SqueezedNorm(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, mid=0, s1=2, s2=2, clip=False):
        self.vmin = vmin # minimum value
        self.mid  = mid  # middle value
        self.vmax = vmax # maximum value
        self.s1=s1; self.s2=s2
        f = lambda x, zero,vmax,s: np.abs((x-zero)/(vmax-zero))**(1./s)*0.5
        self.g = lambda x, zero,vmin,vmax, s1,s2: f(x,zero,vmax,s1)*(x>=zero) - \
                                             f(x,zero,vmin,s2)*(x<zero)+0.5
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        r = self.g(value, self.mid,self.vmin,self.vmax, self.s1,self.s2)
        return np.ma.masked_array(r)


if __name__ == "__main__":
    N = 3
    M = 2
    TRIALS_DIR = f"{TRIALS_DIR}/{N}_{M}"
    x, y, anarchy, stability, all_probs, all_costs = random_simulation(N, M, 0.25, "test_3_2.txt", "test_3_2_key.p", num_simulations=10)
    graph_entropy_and_ne(x, y, anarchy, out_filename="3_2_anarchy.png")
    plt.gcf().clear()
    graph_entropy_and_ne(x, y, stability, title="Price of stability", out_filename="3_2_stability.png")

    with open(f"{TRIALS_DIR}/pickled/simulation.p", "rb") as p_file:
        simulation_data = {"entropy_probs": x, "entropy_costs": y, "probs": all_probs, "costs": all_costs,  "price_anarchy": anarchy, "price_stability": stability}


    