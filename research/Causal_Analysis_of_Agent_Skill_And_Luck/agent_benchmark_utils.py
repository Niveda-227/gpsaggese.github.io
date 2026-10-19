"""
Simulate the winner's curse of picking the best of K agent configs.

The simulation picks the config with the best score on a benchmark and
measures how far that score is above its true resolve rate, and how to
correct it.

The model:
- Config k has a true skill `theta_k ~ Normal(mu_theta, sigma_theta)`
- Task i has a difficulty `b_i ~ Normal(mu_b, sigma_b)`
- Config k and task i have an interaction `u_ki ~ Normal(0, sigma_u)`, so that
  configs are not shifted copies of each other
- `P(resolve) = sigmoid(theta_k - b_i + u_ki)`
- Each seed is an independent Bernoulli draw given that probability

Notation of the white paper (section "Skill-Luck Decomposition"):
- `S_A`: the strategy of agent A, here one config k (agent + scaffold + model)
- `Y`: the `resolved` outcome of config k on one task and one seed
- `Ybar_A = E[Y | S_A, I_A]`: `ybar[k]`, the expected outcome of config k over
  the task population and the seeds (see `compute_true_resolve_rate()`)
- `L_A(Y) = Y - Ybar_A`: the luck component, i.e., which tasks were drawn, the
  interaction `u_ki` of those tasks, and the seed
- The paper also defines a per-event luck score `L_A(E)`, which is not used
  here: the argument `Y` vs `E` distinguishes the two
- `theta_k`, `b_i`, `u_ki` are parameters of this simulation and are not
  symbols of the paper

Tables use the generic columns `config`, `task`, `seed`, `resolved`.

Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.agent_benchmark_utils as rcaoasalabu
"""

import dataclasses
import itertools
import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import tqdm

import helpers.hdbg as hdbg
import helpers.hio as hio
import helpers.hnotebook as hnotebo
import helpers.hprint as hprint

_LOG = logging.getLogger(__name__)

# Bootstrap resampling schemes:
# - `task`: Resample tasks with replacement, keeping all seeds of each task, so
#   that seed outcomes on the same task stay together.
# - `task_seed`: Resample tasks as above, then resample seeds with replacement
#   inside each drawn task (two-stage bootstrap).
BOOTSTRAP_METHODS = ["task", "task_seed"]


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    _LOG.debug("notebook_log=%s", notebook_log.name)
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Model parameters
# #############################################################################


@dataclasses.dataclass(frozen=True)
class SimParams:
    """
    Parameters of one simulated benchmark.

    The defaults are assumptions, not estimates from real data:
    - `sigma_theta=0.2` makes configs close together, which is the realistic
      case for scaffold variants of the same agent
    - `mu_theta=1.0`, `mu_b=0.0`, `sigma_b=2.0`, `sigma_u=1.0` give true
      resolve rates around 64%, with many tasks almost always or almost
      never solved
    """

    n_configs: int = 10
    n_tasks: int = 100
    n_seeds: int = 3
    mu_theta: float = 1.0
    sigma_theta: float = 0.2
    mu_b: float = 0.0
    sigma_b: float = 2.0
    sigma_u: float = 1.0

    def __post_init__(self) -> None:
        """
        Check that sizes are positive and spreads are non-negative.
        """
        hdbg.dassert_lte(1, self.n_configs, msg="Need at least one config")
        hdbg.dassert_lte(1, self.n_tasks, msg="Need at least one task")
        hdbg.dassert_lte(1, self.n_seeds, msg="Need at least one seed")
        hdbg.dassert_lte(
            0,
            min(self.sigma_theta, self.sigma_b, self.sigma_u),
            msg="Standard deviations must be non-negative",
        )


# #############################################################################
# Model draws
# #############################################################################


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the logistic function elementwise.

    :param x: logits
    :return: probabilities in (0, 1)
    """
    _LOG.debug("x.shape=%s", np.shape(x))
    p = 1.0 / (1.0 + np.exp(-x))
    return p


def draw_skills(params: SimParams, rng: np.random.Generator) -> np.ndarray:
    """
    Draw the true skill `theta_k` of each config.

    :param params: simulation parameters
    :param rng: random generator
    :return: skills, shape `(n_configs,)`
    """
    _LOG.debug(hprint.to_str("params"))
    theta = rng.normal(
        params.mu_theta, params.sigma_theta, size=params.n_configs
    )
    return theta


def draw_difficulties(
    n_tasks: int, params: SimParams, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw the difficulty `b_i` of each task from the task population.

    :param n_tasks: number of tasks to draw
    :param params: simulation parameters
    :param rng: random generator
    :return: difficulties, shape `(n_tasks,)`
    """
    _LOG.debug(hprint.to_str("n_tasks"))
    b = rng.normal(params.mu_b, params.sigma_b, size=n_tasks)
    return b


def draw_interactions(
    n_configs: int, n_tasks: int, params: SimParams, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw the config-by-task interaction `u_ki`.

    :param n_configs: number of configs
    :param n_tasks: number of tasks
    :param params: simulation parameters
    :param rng: random generator
    :return: interactions, shape `(n_configs, n_tasks)`
    """
    _LOG.debug(hprint.to_str("n_configs n_tasks"))
    u = rng.normal(0.0, params.sigma_u, size=(n_configs, n_tasks))
    return u


def draw_outcomes(
    p: np.ndarray, n_seeds: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw one Bernoulli outcome per seed for each solve probability.

    :param p: solve probabilities of any shape
    :param n_seeds: number of seeds
    :param rng: random generator
    :return: 0/1 outcomes, shape `p.shape + (n_seeds,)`
    """
    _LOG.debug("p.shape=%s n_seeds=%s", p.shape, n_seeds)
    y = (rng.random(p.shape + (n_seeds,)) < p[..., None]).astype(np.int8)
    return y


def compute_true_resolve_rate(
    theta: np.ndarray, params: SimParams
) -> np.ndarray:
    """
    Compute the true resolve rate `Ybar_A` of each config.

    `Ybar_A` is the expected resolve rate of a config over the task
    population, not over the sampled tasks. For a task drawn from the population, the logit `theta_k - b + u` is
    `Normal(theta_k - mu_b, sigma_b^2 + sigma_u^2)`, so
    `Ybar_k = E[sigmoid(theta_k - mu_b + s Z)]` with
    `s^2 = sigma_b^2 + sigma_u^2` and `Z ~ Normal(0, 1)`. The expectation is
    computed by Gauss-Hermite numerical integration, not by Monte Carlo, so it
    has no sampling noise.

    E.g., `theta_k = mu_b` gives `Ybar_k = 0.5` for any `s`, by symmetry.

    :param theta: skills, shape `(n_configs,)`
    :param params: simulation parameters
    :return: true resolve rates, shape `(n_configs,)`
    """
    _LOG.debug("theta.shape=%s", np.shape(theta))
    # 101 Gauss-Hermite nodes give an error below 1e-12 for the logit spreads
    # used here.
    n_quadrature_nodes = 101
    # `hermegauss()` uses the weight exp(-z^2 / 2), so the weights sum to
    # sqrt(2 pi) and dividing by that gives an expectation under Normal(0, 1).
    nodes, weights = np.polynomial.hermite_e.hermegauss(n_quadrature_nodes)
    weights = weights / np.sqrt(2.0 * np.pi)
    # Integrate the solve probability over the logit of a random task.
    scale = np.sqrt(params.sigma_b**2 + params.sigma_u**2)
    logits = (np.asarray(theta)[:, None] - params.mu_b) + scale * nodes[None, :]
    ybar = sigmoid(logits) @ weights
    return ybar


# #############################################################################
# World
# #############################################################################


@dataclasses.dataclass(frozen=True)
class World:
    """
    One draw of configs, tasks, and outcomes.

    - `theta`: skills, shape `(K,)`
    - `ybar`: true resolve rates `Ybar_A` over the task population, shape `(K,)`
    - `b`: difficulties of the sampled tasks, shape `(N,)`
    - `u`: interactions, shape `(K, N)`
    - `y`: outcomes, shape `(K, N, S)`
    """

    theta: np.ndarray
    ybar: np.ndarray
    b: np.ndarray
    u: np.ndarray
    y: np.ndarray


def draw_world(params: SimParams, rng: np.random.Generator) -> World:
    """
    Draw configs, tasks, interactions, and outcomes for one benchmark run.

    :param params: simulation parameters
    :param rng: random generator
    :return: the drawn world
    """
    _LOG.debug(hprint.to_str("params"))
    # Draw the configs and compute their true resolve rates.
    theta = draw_skills(params, rng)
    ybar = compute_true_resolve_rate(theta, params)
    # Draw the tasks, the interactions, and the seed outcomes.
    b = draw_difficulties(params.n_tasks, params, rng)
    u = draw_interactions(params.n_configs, params.n_tasks, params, rng)
    p = sigmoid(theta[:, None] - b[None, :] + u)
    y = draw_outcomes(p, params.n_seeds, rng)
    world = World(theta=theta, ybar=ybar, b=b, u=u, y=y)
    return world


def outcomes_to_frame(y: np.ndarray) -> pd.DataFrame:
    """
    Convert an outcome array into a long table.

    :param y: outcomes of shape `(K, N, S)`
    :return: table with columns `config`, `task`, `seed`, `resolved`, one row
        per (config, task, seed)
    """
    _LOG.debug("y.shape=%s", y.shape)
    # Build the index of every (config, task, seed) entry.
    n_configs, n_tasks, n_seeds = y.shape
    config, task, seed = np.meshgrid(
        np.arange(n_configs),
        np.arange(n_tasks),
        np.arange(n_seeds),
        indexing="ij",
    )
    # Flatten the index and the outcomes into columns.
    df = pd.DataFrame(
        {
            "config": config.ravel(),
            "task": task.ravel(),
            "seed": seed.ravel(),
            "resolved": y.ravel().astype(int),
        }
    )
    return df


# #############################################################################
# Scores and ranks
# #############################################################################


def compute_scores(y: np.ndarray) -> np.ndarray:
    """
    Compute each config's observed score, i.e., its mean over tasks and seeds.

    :param y: outcomes of shape `(K, N, S)`
    :return: scores, shape `(K,)`
    """
    _LOG.debug("y.shape=%s", y.shape)
    scores = y.mean(axis=(1, 2))
    return scores


def select_winner(scores: np.ndarray) -> int:
    """
    Pick the config with the highest observed score.

    Ties go to the lowest index. Configs are drawn in random order, so this
    does not favor any skill level.

    :param scores: observed scores, shape `(K,)`
    :return: index of the winner
    """
    winner = int(np.argmax(scores))
    _LOG.debug("return=%s", winner)
    return winner


def compute_ranks(values: np.ndarray) -> np.ndarray:
    """
    Rank values from best (1) to worst (K), averaging ranks on ties.

    E.g., `[0.2, 0.5, 0.5]` -> `[3.0, 1.5, 1.5]`.

    :param values: values to rank, shape `(K,)`
    :return: ranks, shape `(K,)`
    """
    _LOG.debug("values.shape=%s", np.shape(values))
    ranks = pd.Series(values).rank(ascending=False, method="average")
    ranks = ranks.to_numpy()
    return ranks


# #############################################################################
# Re-evaluation of the winner
# #############################################################################


def evaluate_on_fresh_seeds(
    world: World,
    config: int,
    n_seeds: int,
    rng: np.random.Generator,
) -> float:
    """
    Rerun one config on the same tasks with new seeds and return its score.

    This removes the seed luck that helped the config win, but not the luck
    of which tasks were in the benchmark.

    :param world: the world the config was selected in
    :param config: index of the config
    :param n_seeds: number of new seeds per task
    :param rng: random generator
    :return: mean outcome over the same tasks and the new seeds
    """
    _LOG.debug(hprint.to_str("config n_seeds"))
    p = sigmoid(world.theta[config] - world.b + world.u[config])
    score = float(draw_outcomes(p, n_seeds, rng).mean())
    _LOG.debug("return=%s", score)
    return score


def evaluate_on_fresh_tasks(
    theta: float,
    params: SimParams,
    rng: np.random.Generator,
) -> float:
    """
    Run one config on new tasks and return its score.

    The config runs on `params.n_tasks` new tasks with `params.n_seeds` seeds
    each. New tasks come with new interactions, so this estimate does not depend on
    anything used to select the config.

    :param theta: skill of the config
    :param params: simulation parameters
    :param rng: random generator
    :return: mean outcome over the new tasks and seeds
    """
    _LOG.debug(hprint.to_str("theta"))
    # Draw new tasks and the interactions of this one config with them.
    b = draw_difficulties(params.n_tasks, params, rng)
    n_configs = 1
    u = draw_interactions(n_configs, params.n_tasks, params, rng)[0]
    # Score the config on the new tasks.
    p = sigmoid(theta - b + u)
    score = float(draw_outcomes(p, params.n_seeds, rng).mean())
    _LOG.debug("return=%s", score)
    return score


# #############################################################################
# Replications
# #############################################################################


def run_replication(
    params: SimParams, rng: np.random.Generator
) -> Dict[str, float]:
    """
    Draw one benchmark, select the winner, and measure what went wrong.

    :param params: simulation parameters
    :param rng: random generator
    :return: metrics of this replication:
        - `curse`: observed score of the winner minus its true `Ybar_A`
        - `is_truly_best`: 1 if the winner has the highest `Ybar_A`, else 0
          - NaN when `sigma_theta == 0`, since then no config is truly best
        - `regret`: best `Ybar_A` minus the winner's `Ybar_A`
        - `rank_error`: mean absolute difference between observed and true
          rank over all configs
        - `fresh_seed_error`: fresh-seed score of the winner minus its
          `Ybar_A`
        - `fresh_task_error`: fresh-task score of the winner minus its
          `Ybar_A`
    """
    _LOG.debug(hprint.to_str("params"))
    # Draw the benchmark and pick the winner.
    world = draw_world(params, rng)
    scores = compute_scores(world.y)
    winner = select_winner(scores)
    # Compare with the true ranking, which is by `Ybar_A` and increasing in
    # `theta`.
    if params.sigma_theta == 0:
        is_truly_best = np.nan
    else:
        is_truly_best = float(winner == int(np.argmax(world.ybar)))
    rank_error = np.abs(compute_ranks(scores) - compute_ranks(world.ybar)).mean()
    # Re-evaluate the winner with the same budget as one config in the
    # original run.
    fresh_seed_score = evaluate_on_fresh_seeds(
        world, winner, params.n_seeds, rng
    )
    fresh_task_score = evaluate_on_fresh_tasks(world.theta[winner], params, rng)
    # Express every error relative to the winner's true resolve rate.
    winner_ybar = world.ybar[winner]
    metrics = {
        "curse": scores[winner] - winner_ybar,
        "is_truly_best": is_truly_best,
        "regret": world.ybar.max() - winner_ybar,
        "rank_error": rank_error,
        "fresh_seed_error": fresh_seed_score - winner_ybar,
        "fresh_task_error": fresh_task_score - winner_ybar,
    }
    return metrics


def run_replications(params: SimParams, n_reps: int, seed: int) -> pd.DataFrame:
    """
    Run `n_reps` independent replications.

    Each replication gets its own generator spawned from `seed`, so a
    replication's result does not depend on the others.

    :param params: simulation parameters
    :param n_reps: number of replications
    :param seed: seed of the whole batch
    :return: one row per replication with the columns of `run_replication()`
    """
    _LOG.debug(hprint.to_str("params n_reps seed"))
    children = np.random.SeedSequence(seed).spawn(n_reps)
    rows = [run_replication(params, np.random.default_rng(c)) for c in children]
    df = pd.DataFrame(rows)
    return df


def summarize_replications(
    df: pd.DataFrame, params: SimParams
) -> Dict[str, float]:
    """
    Average the metrics over replications.

    The summary also has the Monte Carlo standard error of each mean and the
    cost of re-evaluating the winner.

    :param df: output of `run_replications()`
    :param params: parameters used to produce `df`
    :return: for each metric `m`, the keys:
        - `m`: mean over replications
        - `m_se`: Monte Carlo standard error of the mean
        - `extra_runs`: runs needed to re-evaluate the winner, `N * S`
        - `extra_cost_frac`: `extra_runs` over the original `K * N * S` runs
    """
    _LOG.debug("df.shape=%s", df.shape)
    # Average each metric, ignoring the NaNs of `is_truly_best`.
    summary: Dict[str, float] = {}
    for col in df.columns:
        values = df[col].dropna()
        summary[col] = values.mean() if len(values) else np.nan
        summary[f"{col}_se"] = (
            values.std(ddof=1) / np.sqrt(len(values))
            if len(values) > 1
            else np.nan
        )
    # Add the cost of re-evaluating the winner with the same budget.
    extra_runs = params.n_tasks * params.n_seeds
    summary["extra_runs"] = float(extra_runs)
    summary["extra_cost_frac"] = extra_runs / (
        params.n_configs * params.n_tasks * params.n_seeds
    )
    return summary


# #############################################################################
# Bootstrap
# #############################################################################


def bootstrap_scores(
    y: np.ndarray, n_boot: int, method: str, rng: np.random.Generator
) -> np.ndarray:
    """
    Compute bootstrap replicates of one config's score.

    :param y: outcomes of one config, shape `(N, S)`
    :param n_boot: number of bootstrap replicates
    :param method: resampling scheme, one of `BOOTSTRAP_METHODS`
    :param rng: random generator
    :return: bootstrap replicates, shape `(n_boot,)`
    """
    _LOG.debug("y.shape=%s n_boot=%s method=%s", y.shape, n_boot, method)
    # Resample whole tasks.
    n_tasks, n_seeds = y.shape
    task_idx = rng.integers(0, n_tasks, size=(n_boot, n_tasks))
    if method == "task":
        # Resampling whole tasks only needs the per-task means.
        task_means = y.mean(axis=1)
        replicates = task_means[task_idx].mean(axis=1)
    elif method == "task_seed":
        # Draw seeds inside each resampled task, independently per draw.
        seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_tasks, n_seeds))
        replicates = y[task_idx[:, :, None], seed_idx].mean(axis=(1, 2))
    else:
        hdbg.dfatal(f"Unknown bootstrap method='{method}'")
    return replicates


def _compute_percentile_ci(
    y: np.ndarray,
    n_boot: int,
    method: str,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute a percentile bootstrap confidence interval for one config's score.

    :param y: outcomes of one config, shape `(N, S)`
    :param n_boot: number of bootstrap replicates
    :param method: resampling scheme, one of `BOOTSTRAP_METHODS`
    :param rng: random generator
    :param alpha: one minus the confidence level
    :return: lower and upper bound
    """
    _LOG.debug("y.shape=%s n_boot=%s method=%s", y.shape, n_boot, method)
    replicates = bootstrap_scores(y, n_boot, method, rng)
    lo, hi = np.quantile(replicates, [alpha / 2, 1 - alpha / 2])
    ci = (float(lo), float(hi))
    _LOG.debug("return=%s", ci)
    return ci


def compute_cis(
    y: np.ndarray,
    n_boot: int,
    method: str,
    rng: np.random.Generator,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute a percentile bootstrap confidence interval for every config.

    :param y: outcomes, shape `(K, N, S)`
    :param n_boot: number of bootstrap replicates
    :param method: resampling scheme, one of `BOOTSTRAP_METHODS`
    :param rng: random generator
    :param alpha: one minus the confidence level
    :return: one row per config with columns `config`, `score`, `lo`, `hi`
    """
    _LOG.debug("y.shape=%s n_boot=%s method=%s", y.shape, n_boot, method)
    rows = []
    for k in range(y.shape[0]):
        lo, hi = _compute_percentile_ci(y[k], n_boot, method, rng, alpha=alpha)
        rows.append({"config": k, "score": y[k].mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows)
    return df


def run_coverage(
    params: SimParams,
    n_reps: int,
    n_boot: int,
    seed: int,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Measure how often the bootstrap intervals contain the true `Ybar_A`.

    Each replication draws a new world and builds one interval per config and
    method. Coverage is pooled over configs and replications.

    :param params: simulation parameters
    :param n_reps: number of replications
    :param n_boot: number of bootstrap replicates per interval
    :param seed: seed of the whole batch
    :param alpha: one minus the confidence level
    :return: one row per method with columns `method`, `coverage`,
        `coverage_se`, `mean_width`, `n_intervals`
    """
    _LOG.debug(hprint.to_str("params n_reps n_boot seed"))
    # Build the intervals of every replication, config, and method.
    children = np.random.SeedSequence(seed).spawn(n_reps)
    covered: Dict[str, List[np.ndarray]] = {m: [] for m in BOOTSTRAP_METHODS}
    widths: Dict[str, List[np.ndarray]] = {m: [] for m in BOOTSTRAP_METHODS}
    for child in children:
        rng = np.random.default_rng(child)
        world = draw_world(params, rng)
        for method in BOOTSTRAP_METHODS:
            cis = compute_cis(world.y, n_boot, method, rng, alpha=alpha)
            inside = (cis["lo"] <= world.ybar) & (world.ybar <= cis["hi"])
            covered[method].append(inside.to_numpy())
            widths[method].append((cis["hi"] - cis["lo"]).to_numpy())
    # Pool the intervals of all configs and replications.
    rows = []
    for method in BOOTSTRAP_METHODS:
        hits = np.concatenate(covered[method])
        coverage = hits.mean()
        rows.append(
            {
                "method": method,
                "coverage": coverage,
                "coverage_se": np.sqrt(coverage * (1 - coverage) / len(hits)),
                "mean_width": np.concatenate(widths[method]).mean(),
                "n_intervals": len(hits),
            }
        )
    df = pd.DataFrame(rows)
    return df


# #############################################################################
# Sweeps
# #############################################################################


def _setting_seed(seed: int, values: List[int]) -> int:
    """
    Derive a seed for one grid setting from the batch seed and the setting.

    This makes each setting's result independent of which other settings are
    run and in which order.

    :param seed: batch seed
    :param values: the setting, e.g., `[K, N, S]`
    :return: seed of the setting
    """
    _LOG.debug(hprint.to_str("seed values"))
    setting_seed = int(
        np.random.SeedSequence([seed] + values).generate_state(1)[0]
    )
    _LOG.debug("return=%s", setting_seed)
    return setting_seed


def run_curse_sweep(
    base: SimParams,
    n_reps: int,
    seed: int,
    *,
    n_configs_grid: Tuple[int, ...] = (3, 5, 10, 20),
    n_tasks_grid: Tuple[int, ...] = (25, 50, 100, 500),
    n_seeds_grid: Tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """
    Run the winner's curse replications over the (K, N, S) grid.

    :param base: parameters other than K, N, S
    :param n_reps: replications per setting
    :param seed: batch seed
    :param n_configs_grid: values of K
    :param n_tasks_grid: values of N
    :param n_seeds_grid: values of S
    :return: one row per setting with columns `n_configs`, `n_tasks`,
        `n_seeds` and the keys of `summarize_replications()`
    """
    _LOG.debug(hprint.to_str("base n_reps seed"))
    settings = list(
        itertools.product(n_configs_grid, n_tasks_grid, n_seeds_grid)
    )
    rows = []
    for k, n, s in tqdm.tqdm(settings, desc="Curse sweep", ascii=True):
        # Run the replications of one setting with its own seed.
        params = dataclasses.replace(base, n_configs=k, n_tasks=n, n_seeds=s)
        df = run_replications(params, n_reps, _setting_seed(seed, [k, n, s]))
        summary = summarize_replications(df, params)
        rows.append({"n_configs": k, "n_tasks": n, "n_seeds": s, **summary})
        _LOG.debug("K=%s N=%s S=%s curse=%.4f", k, n, s, summary["curse"])
    df_out = pd.DataFrame(rows)
    return df_out


def run_coverage_sweep(
    base: SimParams,
    n_reps: int,
    n_boot: int,
    seed: int,
    *,
    n_tasks_grid: Tuple[int, ...] = (25, 50, 100, 500),
    n_seeds_grid: Tuple[int, ...] = (1, 3, 5, 10),
) -> pd.DataFrame:
    """
    Run bootstrap coverage over the (N, S) grid at K = `base.n_configs`.

    Coverage is measured per config, so K is fixed.

    :param base: parameters other than N and S
    :param n_reps: replications per setting
    :param n_boot: bootstrap replicates per interval
    :param seed: batch seed
    :param n_tasks_grid: values of N
    :param n_seeds_grid: values of S
    :return: one row per (N, S, method) with the columns of `run_coverage()`
    """
    _LOG.debug(hprint.to_str("base n_reps n_boot seed"))
    settings = list(itertools.product(n_tasks_grid, n_seeds_grid))
    dfs = []
    for n, s in tqdm.tqdm(settings, desc="Coverage sweep", ascii=True):
        # Run the coverage of one setting with its own seed.
        params = dataclasses.replace(base, n_tasks=n, n_seeds=s)
        df = run_coverage(
            params, n_reps, n_boot, _setting_seed(seed, [base.n_configs, n, s])
        )
        df.insert(0, "n_seeds", s)
        df.insert(0, "n_tasks", n)
        dfs.append(df)
    df_out = pd.concat(dfs, ignore_index=True)
    return df_out


def run_and_save_sweeps(
    dst_dir: str,
    *,
    n_reps: int = 1000,
    n_coverage_reps: int = 400,
    n_boot: int = 500,
    coverage_k: int = 5,
    seed: int = 0,
) -> None:
    """
    Run both sweeps with the default model and save them as CSV files.

    With the defaults this takes about 5 minutes on one core and writes
    `curse_sweep.csv` and `coverage_sweep.csv` into `dst_dir`.

    :param dst_dir: directory to write the CSV files into
    :param n_reps: replications per setting of the curse sweep
    :param n_coverage_reps: replications per setting of the coverage sweep
    :param n_boot: bootstrap replicates per interval
    :param coverage_k: number of configs K of the coverage sweep
    :param seed: batch seed of both sweeps
    """
    _LOG.debug(hprint.to_str("dst_dir n_reps n_coverage_reps n_boot seed"))
    hio.create_dir(dst_dir, incremental=True)
    base = SimParams()
    # Run and save the curse sweep.
    df_curse = run_curse_sweep(base, n_reps, seed)
    curse_file = os.path.join(dst_dir, "curse_sweep.csv")
    df_curse.to_csv(curse_file, index=False)
    _LOG.info("Saved '%s'", curse_file)
    # Run and save the coverage sweep.
    base_cov = dataclasses.replace(base, n_configs=coverage_k)
    df_cov = run_coverage_sweep(base_cov, n_coverage_reps, n_boot, seed)
    cov_file = os.path.join(dst_dir, "coverage_sweep.csv")
    df_cov.to_csv(cov_file, index=False)
    _LOG.info("Saved '%s'", cov_file)


# #############################################################################
# Plotting
# #############################################################################


# One blue hue stepped light to dark, for ordered values like K or S. Checked
# with a palette validator for monotone lightness and contrast on white.
_ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
_INK = "#2b2b2b"
_MUTED = "#8a8a85"
_GRID = "#e6e6e3"


def describe_sweep_setup(params: SimParams, n_reps: int) -> str:
    """
    Describe the model and the number of replications of a sweep in one line.

    E.g., `mu_theta=1.0, sigma_theta=0.2, mu_b=0.0, sigma_b=2.0, sigma_u=1.0,
    1000 replications per setting`.

    :param params: parameters of the sweep, other than the grid values
    :param n_reps: replications per setting
    :return: text to put under a plot title
    """
    _LOG.debug(hprint.to_str("params n_reps"))
    setup_txt = (
        f"mu_theta={params.mu_theta}, sigma_theta={params.sigma_theta}, "
        f"mu_b={params.mu_b}, sigma_b={params.sigma_b}, "
        f"sigma_u={params.sigma_u}, {n_reps} replications per setting"
    )
    return setup_txt


def _style_axis(ax: plt.Axes, n_tasks: List[int]) -> None:
    """
    Make the grid and spines recessive and use a log x axis for N.

    Ticks go only at the N values of the grid, without minor ticks, so that
    labels do not collide.

    :param ax: axis to style
    :param n_tasks: values of N to put ticks at
    """
    _LOG.debug(hprint.to_str("n_tasks"))
    # Put ticks only at the grid values of N.
    ax.set_xscale("log")
    ax.set_xticks(n_tasks)
    ax.set_xticklabels([str(n) for n in n_tasks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    # Make the grid and the spines recessive.
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_INK, labelsize=8)


def _plot_lines_by(
    ax: plt.Axes,
    df: pd.DataFrame,
    line_col: str,
    y_col: str,
    line_values: List[int],
    label_prefix: str,
) -> None:
    """
    Draw one line per value of `line_col`, with `y_col` against N.

    Each point has a 95% error bar from the column `<y_col>_se`. The error bars come from precomputed Monte Carlo standard errors, which is
    why this uses `matplotlib` instead of `seaborn`.

    :param ax: axis to draw on
    :param df: one row per setting
    :param line_col: column whose values define the lines, e.g., `n_configs`
    :param y_col: column to plot on the y axis
    :param line_values: values of `line_col` to draw, in order
    :param label_prefix: legend label prefix, e.g., `K = `
    """
    _LOG.debug(hprint.to_str("line_col y_col line_values"))
    for color, value in zip(_ORDINAL_BLUES, line_values):
        df_line = df[df[line_col] == value].sort_values("n_tasks")
        ax.errorbar(
            df_line["n_tasks"],
            df_line[y_col],
            yerr=1.96 * df_line[f"{y_col}_se"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            capsize=0,
            label=f"{label_prefix}{value}",
        )


def plot_curse(
    df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the winner's curse and the fresh-task error against N.

    One column per S, one line per K:
    - The top row is the observed score of the winner minus its true `Ybar_A`
    - The bottom row is the same for a re-evaluation of the winner on fresh
      tasks

    :param df: curse sweep, one row per (K, N, S)
    :param setup_txt: set-up of the sweep, from `describe_sweep_setup()`
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug("df.shape=%s figsize=%s", df.shape, figsize)
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    seeds = sorted(df["n_seeds"].unique())
    configs = sorted(df["n_configs"].unique())
    n_tasks = sorted(df["n_tasks"].unique())
    line_col = "n_configs"
    label_prefix = "K = "
    n_rows = 2
    fig, axes = plt.subplots(
        n_rows, len(seeds), figsize=figsize, sharex=True, sharey=True
    )
    rows = [
        ("curse", "Winner's score minus true rate"),
        ("fresh_task_error", "Fresh-task score minus true rate"),
    ]
    for i, (y_col, y_label) in enumerate(rows):
        for j, n_seeds in enumerate(seeds):
            # Draw one panel: one line per K, for one value of S.
            ax = axes[i, j]
            _style_axis(ax, n_tasks)
            ax.axhline(0, color=_MUTED, linewidth=1, linestyle="--")
            df_panel = df[df["n_seeds"] == n_seeds]
            _plot_lines_by(ax, df_panel, line_col, y_col, configs, label_prefix)
            # Label only the outer panels.
            if i == 0:
                ax.set_title(f"S = {n_seeds} seeds", color=_INK, fontsize=10)
            if j == 0:
                ax.set_ylabel(y_label, color=_INK, fontsize=9)
            if i == 1:
                ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0, 0].legend(
        frameon=False, fontsize=8, title="configs", title_fontsize=8
    )
    title = "Winner's curse and fresh-task error against the number of tasks"
    fig.suptitle(f"{title}\n{setup_txt}", color=_INK, fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_prob_best(
    df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the probability that the observed winner is the truly best config.

    One panel per S, one line per K.

    :param df: curse sweep, one row per (K, N, S)
    :param setup_txt: set-up of the sweep, from `describe_sweep_setup()`
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug("df.shape=%s figsize=%s", df.shape, figsize)
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    seeds = sorted(df["n_seeds"].unique())
    configs = sorted(df["n_configs"].unique())
    n_tasks = sorted(df["n_tasks"].unique())
    line_col = "n_configs"
    y_col = "is_truly_best"
    label_prefix = "K = "
    n_rows = 1
    fig, axes = plt.subplots(n_rows, len(seeds), figsize=figsize, sharey=True)
    for j, n_seeds in enumerate(seeds):
        # Draw one panel: one line per K, for one value of S.
        ax = axes[j]
        _style_axis(ax, n_tasks)
        df_panel = df[df["n_seeds"] == n_seeds]
        _plot_lines_by(ax, df_panel, line_col, y_col, configs, label_prefix)
        ax.set_ylim(0, 1)
        ax.set_title(f"S = {n_seeds} seeds", color=_INK, fontsize=10)
        ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0].set_ylabel("P(winner is truly best)", color=_INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, title="configs", title_fontsize=8)
    title = "Probability that the observed winner is the truly best config"
    fig.suptitle(f"{title}\n{setup_txt}", color=_INK, fontsize=10)
    fig.tight_layout()
    plt.show()


def plot_coverage(
    df: pd.DataFrame,
    setup_txt: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot bootstrap CI coverage of the true `Ybar_A` against N.

    One panel per resampling scheme, one line per S, nominal 95% dashed. The
    y axis is not forced to [0, 1], since coverage is between 0.9 and 1 and
    the full range would squash the lines.

    :param df: coverage sweep, one row per (N, S, method)
    :param setup_txt: set-up of the sweep, from `describe_sweep_setup()`
    :param figsize: figure size
        - Default: `plt.rcParams["figure.figsize"]`
    """
    _LOG.debug("df.shape=%s figsize=%s", df.shape, figsize)
    if figsize is None:
        figsize = plt.rcParams["figure.figsize"]
    methods = [
        ("task", "Resample tasks"),
        ("task_seed", "Resample tasks, then seeds"),
    ]
    seeds = sorted(df["n_seeds"].unique())
    n_tasks = sorted(df["n_tasks"].unique())
    line_col = "n_seeds"
    y_col = "coverage"
    label_prefix = "S = "
    nominal_coverage = 0.95
    n_rows = 1
    fig, axes = plt.subplots(n_rows, len(methods), figsize=figsize, sharey=True)
    for ax, (method, panel_title) in zip(axes, methods):
        # Draw one panel: one line per S, for one resampling scheme.
        _style_axis(ax, n_tasks)
        ax.axhline(nominal_coverage, color=_MUTED, linewidth=1, linestyle="--")
        df_panel = df[df["method"] == method]
        _plot_lines_by(ax, df_panel, line_col, y_col, seeds, label_prefix)
        ax.set_title(panel_title, color=_INK, fontsize=10)
        ax.set_xlabel("N tasks", color=_INK, fontsize=9)
    axes[0].set_ylabel("Coverage of true rate (95% CI)", color=_INK, fontsize=9)
    axes[1].legend(
        frameon=False,
        fontsize=8,
        title="seeds",
        title_fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    title = "Coverage of 95% bootstrap intervals of one config's true rate"
    fig.suptitle(f"{title}\n{setup_txt}", color=_INK, fontsize=10)
    fig.tight_layout()
    plt.show()
