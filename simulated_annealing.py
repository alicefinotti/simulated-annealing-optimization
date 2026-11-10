from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

## Stochastically determine whether to acccept a move according to the
## Metropolis rule (valid for symmetric proposals)
def accept(delta_c, beta):
    ## If the cost doesn't increase, we always accept
    if delta_c <= 0:
        return True
    ## If the cost increases and beta is infinite, we always reject
    if beta == np.inf:
        return False
    ## Otherwise the probability is going to be somwhere between 0 and 1
    p = np.exp(-beta * delta_c)
    ## Returns True with probability p
    return np.random.rand() < p

## The simulated annealing generic solver.
## Assumes that the proposals are symmetric.
## The `probl` object must implement these methods:
##    init_config()               # returns None [changes internal config]
##    cost()                      # returns a real number
##    propose_move()              # returns a (problem-dependent) move - must be symmetric!
##    compute_delta_cost(move)    # returns a real number
##    accept_move(move)           # returns None [changes internal config]
##    copy()                      # returns a new, independent opbject
## NOTE: The default beta0 and beta1 are arbitrary.
def simann(probl,
           anneal_steps = 10, mcmc_steps = 100,
           beta0 = 0.1, beta1 = 10.0,
           seed = None, debug_delta_cost = False):
    ## Optionally set up the random number generator state
    if seed is not None:
        np.random.seed(seed)

    # Set up the list of betas.
    # First allocate an array with the required number of steps
    beta_list = np.zeros(anneal_steps)
    # All but the last one are evenly spaced between beta0 and beta1 (included)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    # The last one is set to infinty
    beta_list[-1] = np.inf

    # Set up the initial configuration, compute and print the initial cost
    probl.init_config()
    c = probl.cost()
    print(f"initial cost = {c}")

    ## Keep the best cost seen so far, and its associated configuration.
    best = probl.copy()
    best_c = c

    # Main loop of the annaling: Loop over the betas
    for beta in beta_list:
        ## At each beta, we want to record the acceptance rate, so we need a
        ## counter for the number of accepted moves
        accepted = 0
        # For each beta, perform a number of MCMC steps
        for t in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            ## Optinal (expensive) check that `compute_delta_cost` works
            if debug_delta_cost:
                probl_copy = probl.copy()
                probl_copy.accept_move(move)
                assert abs(c + delta_c - probl_copy.cost()) < 1e-10
            ## Metropolis rule
            if accept(delta_c, beta):
                probl.accept_move(move)
                c += delta_c
                accepted += 1
                if c <= best_c:
                    best_c = c
                    best = probl.copy()
        print(f"acc.rate={accepted/mcmc_steps} beta={beta} c={c} [best={best_c}]")

    ## Return the best instance
    print(f"final cost = {best_c}")
    return best



def generate_data(n, seed):
    if type(seed) != str:
        raise TypeError("The seed should be the string representing your ID")
    if type(n) != int:
        raise TypeError("The dimension of the problem should be an integer value")
    if n%2 != 0:
        raise ValueError("The dimension of the problem should be even!!")
    
    aggregate_counter = 0
    for char in seed:
        aggregate_counter += int(char)
    
    aggregate_counter = 10*aggregate_counter
    marginal_diff = int(aggregate_counter)%n 
    bsize = 12 + int(np.floor(n**(1/2)))
    if (bsize%2)!=0:
        bsize=bsize-1
    
    if (marginal_diff<bsize+4):
        marginal_diff = bsize+4  
    if (marginal_diff > (n-bsize-4)):
        marginal_diff = n-bsize-4
    if ((marginal_diff > n/2 - bsize-5) and (marginal_diff < n/2+bsize+5)):
        marginal_diff = int(n/2-bsize-5)
    
    c_marginal_diff = n-marginal_diff
    
    b1 = -100 + np.random.randn(bsize, bsize)
    b2 = -50 + np.random.randn(bsize, bsize)
    

    f_values = np.zeros((n,n), dtype=np.float32)
    f_values = 1+ 0.05*np.random.randn(n,n)
    f_values[marginal_diff-bsize//2:marginal_diff+bsize//2, c_marginal_diff-bsize//2:c_marginal_diff+bsize//2] = b1
    f_values[c_marginal_diff-bsize//2:c_marginal_diff+bsize//2, marginal_diff-bsize//2:marginal_diff+bsize//2] = b2
    return f_values


class GridProblem:
    """Minimize C[x, y] using symmetric diagonal ±1 moves with wraparound."""

    def __init__(self, C, fixed_start=None):
        C = np.asarray(C)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
            raise ValueError("C must be a square matrix of size at least 2")
        self.C = C  # shared read-only lookup; copies keep a reference to it
        self.n = len(C)
        self.fixed_start = fixed_start
        self.x = 0
        self.y = 0
        self.init_config()

    def init_config(self):
        if self.fixed_start is None:
            self.x, self.y = map(int, np.random.randint(0, self.n, size=2))
        else:
            x, y = self.fixed_start
            if not (0 <= x < self.n and 0 <= y < self.n):
                raise ValueError("fixed_start must be within grid boundaries")
            self.x, self.y = int(x), int(y)

    def cost(self):
        return float(self.C[self.x, self.y])

    def propose_move(self):
        # The original proposal design changes BOTH coordinates by ±1.
        dx = int(np.random.choice([-1, 1]))
        dy = int(np.random.choice([-1, 1]))
        return ((self.x + dx) % self.n, (self.y + dy) % self.n)

    def compute_delta_cost(self, move):
        return float(self.C[move] - self.C[self.x, self.y])

    def accept_move(self, move):
        self.x, self.y = map(int, move)

    def copy(self):
        # Unlike deepcopy of a dense n×n matrix, copy only the mutable state.
        # This is valid because the landscape is never modified by the solver.
        obj = object.__new__(GridProblem)
        obj.C = self.C
        obj.n = self.n
        obj.fixed_start = self.fixed_start
        obj.x, obj.y = self.x, self.y
        return obj


@dataclass
class RunResult:
    method: str
    initial_cost: float
    final_cost: float
    best_cost: float
    best_position: tuple
    first_hit_step: int | None
    betas: np.ndarray
    costs: np.ndarray
    best_costs: np.ndarray
    acceptance_probabilities: np.ndarray
    accepted: np.ndarray


def run_with_metrics(probl, method="annealing", anneal_steps=50, mcmc_steps=100,
                     beta0=0.01, beta1=2.0, seed=None, target=-95.0):
    if method not in ("annealing", "greedy", "random_walk"):
        raise ValueError("Unknown method")
    if anneal_steps < 2 or mcmc_steps < 1:
        raise ValueError("anneal_steps must be >= 2, mcmc_steps >= 1")
    if seed is not None:
        np.random.seed(seed)

    # Exactly the linear-in-beta schedule used in SimAnn_ok.py.
    beta_list = np.zeros(anneal_steps)
    beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
    beta_list[-1] = np.inf
    if method == "greedy":
        beta_list[:] = np.inf
    elif method == "random_walk":
        beta_list[:] = 0.0

    probl.init_config()
    current = probl.cost()
    initial = current
    best = probl.copy()
    best_c = current
    first_hit = 0 if best_c <= target else None
    costs, best_costs, probabilities, accepted_flags, betas = [], [], [], [], []
    step = 0
    for beta in beta_list:
        for _ in range(mcmc_steps):
            move = probl.propose_move()
            delta_c = probl.compute_delta_cost(move)
            probability = (1.0 if delta_c <= 0 else
                           0.0 if beta == np.inf else float(np.exp(-beta * delta_c)))
            accepted = accept(delta_c, beta)  # unchanged original function
            if accepted:
                probl.accept_move(move)
                current += delta_c
                if current <= best_c:
                    best_c = current
                    best = probl.copy()
            step += 1
            if first_hit is None and best_c <= target:
                first_hit = step
            costs.append(current)
            best_costs.append(best_c)
            probabilities.append(probability)
            accepted_flags.append(accepted)
            betas.append(beta)

    return RunResult(method, initial, current, best_c, (best.x, best.y), first_hit,
                     np.asarray(betas), np.asarray(costs), np.asarray(best_costs),
                     np.asarray(probabilities), np.asarray(accepted_flags))



METHODS = ('annealing', 'greedy', 'random_walk')


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='100,200,500,1000')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--anneal-steps', type=int, default=50)
    parser.add_argument('--mcmc-steps', type=int, default=100)
    parser.add_argument('--beta0', type=float, default=0.01)
    parser.add_argument('--beta1', type=float, default=2.0)
    parser.add_argument('--position-seed', default='1234567', help='Public example string, not a personal ID')
    parser.add_argument('--random-seed', type=int, default=2026)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    sizes = [int(s) for s in args.sizes.split(',')]
    if any(n < 100 or n % 2 for n in sizes) or args.trials < 1:
        parser.error('Choose even sizes >= 100 and at least one trial')
    args.output.mkdir(exist_ok=True, parents=True)

    rows = []
    example = {}
    demo_n = 200 if 200 in sizes else sizes[0]
    for n in sizes:
        # Original generate_data.py is unchanged. As its seed only controls
        # basin locations, set NumPy's seed explicitly for reproducible noise.
        np.random.seed(args.random_seed + n)
        C = generate_data(n, args.position_seed)
        true_min = float(C.min())
        if n == demo_n:
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(C, interpolation='nearest', origin='lower')
            fig.colorbar(im, ax=ax, label='Objective value')
            ax.set(title=f'Optimization landscape (n = {n})', xlabel='y', ylabel='x')
            fig.tight_layout()
            fig.savefig(args.output / 'landscape.png', dpi=150)
            plt.close(fig)

        for trial in range(args.trials):
            # All methods get the same start within each trial.
            init_rng = np.random.default_rng(args.random_seed + 100_000 * n + trial)
            start = (int(init_rng.integers(n)), int(init_rng.integers(n)))
            for j, method in enumerate(METHODS):
                problem = GridProblem(C, fixed_start=start)
                result = run_with_metrics(problem, method=method,
                          anneal_steps=args.anneal_steps, mcmc_steps=args.mcmc_steps,
                          beta0=args.beta0, beta1=args.beta1,
                          seed=args.random_seed + 10000 * n + 10 * trial + j)
                rows.append(dict(n=n, trial=trial+1, method=method,
                                 true_minimum=round(true_min, 4),
                                 start_x=start[0], start_y=start[1],
                                 initial_cost=round(result.initial_cost, 4),
                                 best_cost=round(result.best_cost, 4),
                                 final_cost=round(result.final_cost, 4),
                                 deep_basin_hit=int(result.best_cost <= -95.0),
                                 first_hit_step=result.first_hit_step if result.first_hit_step is not None else '',
                                 acceptance_rate=round(float(result.accepted.mean()), 4)))
                if n == demo_n and trial == 0:
                    example[method] = result
        print(f'Completed n={n}, true minimum={true_min:.2f}')

    summary = []
    for n in sizes:
        for method in METHODS:
            group = [r for r in rows if r['n'] == n and r['method'] == method]
            hits = sum(r['deep_basin_hit'] for r in group)
            summary.append(dict(n=n, method=method, trials=len(group), hits=hits,
                                success_rate=round(hits/len(group), 3),
                                mean_best_cost=round(float(np.mean([r['best_cost'] for r in group])), 3)))
    write_csv(args.output / 'runs.csv', rows)
    write_csv(args.output / 'summary.csv', summary)
    (args.output / 'config.json').write_text(json.dumps(vars(args) | {'output': str(args.output)}, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        sub = [r for r in summary if r['method'] == method]
        ax.plot([r['n'] for r in sub], [r['success_rate'] for r in sub], marker='o', label=method.replace('_', ' '))
    ax.set(xlabel='Grid width n', ylabel='Proportion reaching cost <= -95',
           ylim=(-0.03, 1.03), title='Deep-basin hit rate by grid size')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'success_by_size.png', dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for method, r in example.items():
        ax.plot(r.best_costs, label=method.replace('_', ' '), linewidth=1.5)
    ax.set(xlabel='Proposed moves', ylabel='Best cost observed',
           title=f'Optimization trajectories (n={demo_n})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'example_convergence.png', dpi=150)
    plt.close(fig)

    ann = example['annealing']
    p = [float(ann.acceptance_probabilities[i:i+args.mcmc_steps].mean())
         for i in range(0, len(ann.accepted), args.mcmc_steps)]
    a = [float(ann.accepted[i:i+args.mcmc_steps].mean())
         for i in range(0, len(ann.accepted), args.mcmc_steps)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, label='Mean proposed acceptance probability')
    ax.plot(a, label='Observed fraction accepted')
    ax.set(xlabel='Annealing stage', ylabel='Acceptance (0 to 1)',
           ylim=(-0.03, 1.03), title=f'Move acceptance by stage (n={demo_n})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / 'acceptance_by_stage.png', dpi=150)
    plt.close(fig)

    print('\nNear-deep-basin hits (new benchmark, not historical report):')
    for row in summary:
        print(f"n={row['n']:4d} {row['method']:12s}: {row['hits']}/{row['trials']}")


if __name__ == '__main__':
    main()

