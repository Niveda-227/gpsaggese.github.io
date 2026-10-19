# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Agent benchmark API
#
# - This notebook explores `agent_benchmark_utils`, a simulation of picking the
#   best of K agent configs on a benchmark of N tasks run with S seeds each
# - The running example is a small benchmark with 4 configs, 30 tasks, and 3
#   seeds, small enough to look at every array
# - The pedagogical arc:
#   - Primitives: the model parameters, the draws, and the true resolve rate
#     $\bar{Y}_A$
#   - Composition: one benchmark run, the winner, and its winner's curse
#   - Patterns: re-evaluating the winner, replications, and bootstrap intervals

# %%
# %load_ext autoreload
# %autoreload 2

import logging

import numpy as np

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook

import agent_benchmark_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# ## Library Overview
#
# - **What problem it solves**: a benchmark score mixes skill and luck, so the
#   config with the best score is not always the best config, and its score is
#   too high on average (the winner's curse)
# - **Key abstraction**: a `World`, i.e., one draw of config skills, tasks, and
#   seed outcomes, from which scores, winners, and errors are computed
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `SimParams(...)` | Sizes K, N, S and the spreads of the model | Frozen dataclass |
# | `draw_world(params, rng)` | One benchmark run | Returns a `World` |
# | `World.ybar` | True resolve rate $\bar{Y}_A$ of each config | Shape `(K,)` |
# | `World.y` | 0/1 outcomes | Shape `(K, N, S)` |
# | `compute_scores(y)` | Observed score of each config | Mean over tasks and seeds |
# | `run_replications(params, n_reps, seed)` | Many independent runs | One row of metrics per run |
# | `compute_cis(y, n_boot, method, rng)` | Bootstrap interval per config | Methods `task`, `task_seed` |
#
# - **Key classes**:
#   - `SimParams`: the parameters of the simulated benchmark
#   - `World`: the arrays of one benchmark run
# - **Notation** of the white paper (section "Skill-Luck Decomposition"):
#   - $S_A$: the strategy of agent $A$, here one config $k$
#   - $Y$: the `resolved` outcome of a config on one task and one seed
#   - $\bar{Y}_A = E[Y \mid S_A, I_A]$: the true resolve rate `ybar` of a config
#   - $L_A(Y) = Y - \bar{Y}_A$: the luck component of the outcome
#   - $\theta_k$, $b_i$, $u_{ki}$ are parameters of this simulation, not
#     symbols of the paper

# %% [markdown]
# # Part 1: The model

# %% [markdown]
# ## Cell 1.1: Model parameters
#
# - Config $k$ has skill $\theta_k \sim N(\mu_\theta, \sigma_\theta^2)$
# - Task $i$ has difficulty $b_i \sim N(\mu_b, \sigma_b^2)$
# - Config $k$ and task $i$ have an interaction $u_{ki} \sim N(0, \sigma_u^2)$
# - Each seed is a Bernoulli draw with
#   $P(\text{resolve}) = \text{sigmoid}(\theta_k - b_i + u_{ki})$

# %%
# Build the parameters of the running example.
params = utils.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
print("params=", params)
# Outcome: the sizes are set, and the spreads keep their defaults.

# %%
# Show the default parameters, which are assumptions and not estimates.
print("utils.SimParams()=", utils.SimParams())
# Outcome: 10 configs, 100 tasks, 3 seeds, with configs close together
# (`sigma_theta=0.2`) and tasks far apart (`sigma_b=2.0`).

# %% [markdown]
# ## Cell 1.2: Drawing skills, difficulties, and interactions

# %%
# Draw the skill of each config.
seed = 0
rng = np.random.default_rng(seed)
theta = utils.draw_skills(params, rng)
print("theta=", theta)
# Outcome: 4 skills close to `mu_theta=1.0`.

# %%
# Draw the difficulty of each task.
b = utils.draw_difficulties(params.n_tasks, params, rng)
print("b.shape=", b.shape)
print("b[:5]=", b[:5])
# Outcome: 30 difficulties spread much more widely than the skills.

# %%
# Draw the config-by-task interactions.
u = utils.draw_interactions(params.n_configs, params.n_tasks, params, rng)
print("u.shape=", u.shape)
# Outcome: one interaction per (config, task), so a config can be lucky on
# some tasks and not on others.

# %%
# Draw the seed outcomes from the solve probabilities.
p = utils.sigmoid(theta[:, None] - b[None, :] + u)
y = utils.draw_outcomes(p, params.n_seeds, rng)
print("y.shape=", y.shape)
# Outcome: a 0/1 array of shape (K, N, S).

# %% [markdown]
# ## Cell 1.3: The true resolve rate
#
# - $\bar{Y}_A$ is the expected outcome of a config over the whole task
#   population and the seeds, not over the 30 tasks drawn
# - It is computed by numerical integration, so it has no sampling noise

# %%
# Compute the true resolve rate of each config.
ybar = utils.compute_true_resolve_rate(theta, params)
print("ybar=", ybar)
# Outcome: rates around 0.64, in the same order as `theta`.

# %%
# Check the symmetry case: `theta = mu_b` gives exactly 0.5.
ybar_sym = utils.compute_true_resolve_rate(np.array([params.mu_b]), params)
print("ybar_sym=", ybar_sym)
# Outcome: 0.5.

# %% [markdown]
# ## Cell 1.4: One benchmark run

# %%
# Draw a whole benchmark run in one call.
world = utils.draw_world(params, np.random.default_rng(seed))
print("world.ybar=", world.ybar)
print("world.y.shape=", world.y.shape)
# Outcome: the same arrays as in the cells above, bundled in a `World`.

# %%
# Convert the outcomes into the generic long table.
df_outcomes = utils.outcomes_to_frame(world.y)
display(df_outcomes.head())
# Outcome: one row per (config, task, seed) with columns `config`, `task`,
# `seed`, `resolved`.

# %%
hintros.print_obj_info(utils.draw_world)

# %% [markdown]
# # Part 2: Scores and the winner's curse

# %% [markdown]
# ## Cell 2.1: Observed scores and the winner

# %%
# Compute the observed score of each config.
scores = utils.compute_scores(world.y)
print("scores=", scores)
# Outcome: the mean over tasks and seeds, which mixes skill $\bar{Y}_A$ and
# luck $L_A(Y)$.

# %%
# Pick the config with the best observed score.
winner = utils.select_winner(scores)
print("winner=", winner)
# Outcome: the index of the highest score.

# %%
# Check whether the winner is also the truly best config.
truly_best = int(np.argmax(world.ybar))
print("truly_best=", truly_best)
# Outcome: with configs this close, the two can differ. Here the winner is
# config 1 and the truly best is config 2.

# %% [markdown]
# ## Cell 2.2: Ranks and the curse

# %%
# Compare the observed ranks with the true ranks.
print("observed ranks=", utils.compute_ranks(scores))
print("true ranks=", utils.compute_ranks(world.ybar))
# Outcome: rank 1 is the best. Here the observed winner, config 1, has the
# worst true rate of the 4.

# %%
# Measure the winner's curse of this run.
curse = scores[winner] - world.ybar[winner]
print("curse=", curse)
# Outcome: the winner's observed score minus its true rate. One run is noisy;
# Part 4 averages many runs.

# %% [markdown]
# # Part 3: Re-evaluating the winner

# %% [markdown]
# ## Cell 3.1: Fresh seeds on the same tasks
#
# - Rerunning the winner with new seeds removes the seed luck
# - It keeps the luck of which tasks were in the benchmark, including the
#   interactions $u_{ki}$ of the winner with those tasks

# %%
# Rerun the winner on the same tasks with new seeds.
fresh_seed = 1
fresh_seed_score = utils.evaluate_on_fresh_seeds(
    world, winner, params.n_seeds, np.random.default_rng(fresh_seed)
)
print("fresh_seed_score=", fresh_seed_score)
# Outcome: a new score for the winner on the same 30 tasks, lower than its
# observed score but still above its true rate.

# %% [markdown]
# ## Cell 3.2: Fresh tasks
#
# - Running the winner on new tasks does not reuse anything that selected it

# %%
# Run the winner on new tasks with the same budget.
fresh_task_seed = 2
fresh_task_score = utils.evaluate_on_fresh_tasks(
    world.theta[winner], params, np.random.default_rng(fresh_task_seed)
)
print("fresh_task_score=", fresh_task_score)
# Outcome: a score that is unbiased for the winner's true rate. In this one
# run it happens to equal the fresh-seed score; Part 4 averages many runs.

# %% [markdown]
# # Part 4: Replications

# %% [markdown]
# ## Cell 4.1: One replication

# %%
# Run one replication and collect its metrics.
metrics = utils.run_replication(params, np.random.default_rng(seed))
print("metrics=", metrics)
# Outcome: the curse, whether the winner is truly best, the regret, the rank
# error, and the two re-evaluation errors.

# %% [markdown]
# ## Cell 4.2: Many replications

# %%
# Run 500 independent replications.
n_reps = 500
df_reps = utils.run_replications(params, n_reps, seed)
display(df_reps.head())
# Outcome: one row of metrics per replication.

# %%
# Average the metrics and add Monte Carlo standard errors.
summary = utils.summarize_replications(df_reps, params)
print("summary=", summary)
# Outcome:
# - `curse` is positive, about 3.9 points
# - `fresh_seed_error` is between 0 and the curse
# - `fresh_task_error` is within 2 standard errors of 0

# %%
hintros.print_obj_info(utils.run_replication)

# %% [markdown]
# # Part 5: Bootstrap intervals

# %% [markdown]
# ## Cell 5.1: Bootstrap replicates of one score
#
# - `task`: resample tasks, keeping all the seeds of each task together
# - `task_seed`: resample tasks, then resample seeds inside each drawn task

# %%
# Print the available resampling schemes.
print("utils.BOOTSTRAP_METHODS=", utils.BOOTSTRAP_METHODS)
# Outcome: `task` and `task_seed`.

# %%
# Compute bootstrap replicates of the score of config 0.
n_boot = 1000
method = "task"
replicates = utils.bootstrap_scores(
    world.y[0], n_boot, method, np.random.default_rng(seed)
)
print("replicates.std()=", replicates.std())
# Outcome: the bootstrap standard error of one config's score.

# %% [markdown]
# ## Cell 5.2: One interval per config

# %%
# Compute a 95% interval for every config.
df_cis = utils.compute_cis(world.y, n_boot, method, np.random.default_rng(seed))
df_cis["ybar"] = world.ybar
display(df_cis)
# Outcome: one interval per config, next to the true rate it should contain.

# %% [markdown]
# ## Cell 5.3: Coverage of the intervals

# %%
# Measure how often each scheme covers the true rate.
n_coverage_reps = 50
n_coverage_boot = 200
df_cov = utils.run_coverage(params, n_coverage_reps, n_coverage_boot, seed)
display(df_cov)
# Outcome: coverage of both schemes over 200 intervals, close to 95%. The
# example notebook shows the full grid.

# %% [markdown]
# ## Summary: The Mental Model
#
# - A benchmark score is the true resolve rate $\bar{Y}_A$ plus luck
#   $L_A(Y)$, and the luck comes from the tasks drawn, the config-by-task
#   interactions, and the seeds
# - Picking the best observed score selects for luck, so the winner's score is
#   too high on average: the winner's curse
# - Rerunning the winner on the same tasks with new seeds removes only part of
#   the curse; running it on fresh tasks removes all of it
# - Bootstrap intervals that resample tasks cover the true rate close to 95% of
#   the time; also resampling seeds inside each task counts seed noise twice
#   when seeds are independent, so its intervals are wider than needed
