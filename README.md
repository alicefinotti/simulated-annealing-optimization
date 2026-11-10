# Simulated Annealing for Discrete Optimization

**Python · Stochastic optimization · Metropolis algorithm · Computational experiments**

## Overview

This project investigates **simulated annealing on a two-dimensional discrete grid**, focusing on the influence of temperature schedules, initial conditions, and grid size on convergence. It also compares annealing with greedy search and random-walk behavior on the same synthetic objective function.

The optimization landscape is mostly flat, with two small low-cost regions (approximately **−100** and **−50**). The original study examines why these regions become increasingly difficult to locate as the search space grows.

## Methodology

Candidate solutions are proposed by moving each coordinate by ±1, with periodic boundaries. Moves are accepted using the Metropolis rule:

$$
P(\text{accept}) = \min\left(1, \exp(-\beta\Delta C)\right), \qquad \beta=1/T.
$$

The experiments track solution quality and acceptance rates under varying temperature schedules and grid sizes. See the **[technical report](report.pdf)** for the original analysis, figures and discussion.

## Findings

- In the original experiments, the success rate decreased from **90% for a grid of width 100** to **0% at width 5,000** under the tested settings (20 trials per size).
- Exploration at low inverse temperature helped discover distant low-cost regions, while greedy behavior was vulnerable to the initial position.
- For this specific landscape, the near-random search regime performed well compared with the tested annealing and greedy configurations; the result should not be generalized to other optimization problems.


