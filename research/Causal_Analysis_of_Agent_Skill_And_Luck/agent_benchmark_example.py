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
# # Agent benchmark example: the winner's curse
#
# - This notebook measures the winner's curse when picking the best of K agent
#   configs on a benchmark of N tasks with S seeds per task
# - It runs a small sweep live, then reads the full sweep saved in
#   `agent_benchmark_results/` (1000 replications per setting, seed 0)
# - The pedagogical arc:
#   - Run a sweep over (K, N, S) and read its table
#   - See how big the curse is and how often the winner is truly best
#   - Compare the two ways to re-evaluate the winner
#   - Check how often bootstrap intervals cover the true rate
#   - List the assumptions that could make a real benchmark behave differently

# %%
# %load_ext autoreload
# %autoreload 2

import logging
import os

import pandas as pd

# %%
import helpers.hintrospection as hintros
import helpers.hnotebook as hnotebook
import helpers.hprint as hprint

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
# # Part 1: Running a sweep

# %% [markdown]
# ## Cell 1.1: A small sweep
#
# - Each setting (K, N, S) runs independent replications of one benchmark run
# - Each setting has its own seed derived from the batch seed, so a setting gives
#   the same result whatever other settings run with it

# %%
# Run a small sweep with 200 replications per setting.
base = utils.SimParams()
n_reps_small = 200
seed = 0
df_small = utils.run_curse_sweep(
    base,
    n_reps_small,
    seed,
    n_configs_grid=(3, 10),
    n_tasks_grid=(25, 100),
    n_seeds_grid=(1, 3),
)
display(df_small[["n_configs", "n_tasks", "n_seeds", "curse", "curse_se"]])
# Outcome: 8 settings, with the largest curse at K = 10, N = 25, S = 1.

# %% [markdown]
# ## Cell 1.2: Regenerating the full sweep
#
# - The full sweep has 64 settings for the curse and 16 for coverage, and takes
#   about 5 minutes on one core
# - Set `rerun_sweeps = True` to regenerate the saved CSV files

# %%
# Set the parameters that produced the saved results.
results_dir = "agent_benchmark_results"
n_reps = 1000
n_coverage_reps = 400
n_boot = 500
coverage_k = 5
print("n_reps=", n_reps)
print("n_coverage_reps=", n_coverage_reps)
# Outcome: the same values as the defaults of `run_and_save_sweeps()`.

# %%
# Regenerate the saved results only when asked.
rerun_sweeps = False
if rerun_sweeps:
    utils.run_and_save_sweeps(
        results_dir,
        n_reps=n_reps,
        n_coverage_reps=n_coverage_reps,
        n_boot=n_boot,
        coverage_k=coverage_k,
        seed=seed,
    )
print("rerun_sweeps=", rerun_sweeps)
# Outcome: the saved CSV files are used as they are.

# %%
hintros.print_obj_info(utils.run_and_save_sweeps)

# %% [markdown]
# ## Cell 1.3: The full sweep

# %%
# Load the full curse sweep.
df_curse = pd.read_csv(os.path.join(results_dir, "curse_sweep.csv"))
print("df_curse.shape=", df_curse.shape)
display(df_curse.head())
# Outcome: 64 rows, one per (K, N, S), with each metric and its Monte Carlo
# standard error.

# %%
# Load the full coverage sweep.
df_cov = pd.read_csv(os.path.join(results_dir, "coverage_sweep.csv"))
print("df_cov.shape=", df_cov.shape)
display(df_cov.head())
# Outcome: 32 rows, one per (N, S, method), at K = 5.

# %% [markdown]
# # Part 2: The winner's curse

# %% [markdown]
# ## Cell 2.1: Size of the curse
#
# **Goal**
# - See how far the winner's observed score is above its true rate
#   $\bar{Y}_A$
# - Compare more candidates K against more tasks N and more seeds S

# %% [markdown]
# **Description**
# - Inputs
#   - `df_curse`: the full curse sweep
# - Panels
#   - Top row: the winner's score minus its true rate, against N, one line per K
#   - Bottom row: the same for a re-evaluation of the winner on fresh tasks
#   - Columns: S = 1, 3, 5, 10 seeds

# %%
# Describe the set-up of the sweep, to show it under each plot title.
setup_txt = utils.describe_sweep_setup(base, n_reps)
print("setup_txt=", setup_txt)
# Outcome: the model parameters and the number of replications.

# %%
# Plot the curse and the fresh-task error against N.
utils.plot_curse(df_curse, setup_txt, figsize=(13, 6.5))

# %% [markdown]
# **Guided usage**
# - Look at the top row
#   - Observe every point above the dashed zero line: the curse is positive in
#     all 64 settings
#   - Observe the lines go down as N grows, in every panel
#   - Observe the lines move up from K = 3 to K = 20, except at N = 500 and
#     S = 10, where K = 10 and K = 20 are within noise of each other
# - Compare the same panel across columns
#   - Observe that more seeds lower the curse less than more tasks do
# - Look at the bottom row
#   - Observe the lines close to the zero line: 59 of the 64 error bars cross
#     zero, so fresh tasks remove the curse

# %% [markdown]
# **Implementation** `plot_curse(df, setup_txt, figsize=None)`
# - Draws a 2 by 4 grid of panels, one column per S
# - Draws one line per K with 95% error bars from the precomputed standard
#   errors, via `_plot_lines_by()`
# - Uses a log x axis with ticks at the N values of the grid, via `_style_axis()`
# - Puts the set-up of the sweep under the title

# %%
hintros.print_obj_info(utils.plot_curse)

# %% [markdown]
# ## Cell 2.2: The curse in numbers
#
# - Some settings from the sweep, in points of resolve rate:
#   - K = 10, N = 100, S = 3: 2.8 points too high
#   - K = 20, N = 25, S = 1: 13.3 points too high
#   - K = 3, N = 500, S = 10: 0.2 points too high
# - At K = 10 and S = 3, going from 25 to 500 tasks cuts the curse from 7.0 to
#   0.6 points
# - At K = 10 and N = 25, going from 1 to 10 seeds cuts it from 10.7 to 4.9
#   points

# %%
# Show the curse at K = 10 for every (N, S), in points.
n_configs = 10
df_k10 = df_curse[df_curse["n_configs"] == n_configs]
print("kept=", hprint.perc(len(df_k10), len(df_curse)))
display(100 * df_k10.pivot(index="n_tasks", columns="n_seeds", values="curse"))
# Outcome: 16 of the 64 rows are kept, and the curse falls along both axes,
# faster along N.

# %% [markdown]
# ## Cell 2.3: Tasks against seeds at the same budget
#
# - The grid has 8 groups of settings with the same K and the same runs per
#   config N S, i.e., 250 or 500
# - In all 8 groups, the setting with more tasks has the smaller curse
#   - E.g., at K = 10, N = 500, S = 1 gives 1.5 points and N = 50, S = 10 gives
#     3.0 points

# %%
# Compare settings with the same K and the same runs per config.
df_budget = df_curse.assign(runs=df_curse["n_tasks"] * df_curse["n_seeds"])
mask = df_budget.duplicated(["n_configs", "runs"], keep=False)
df_budget = df_budget[mask].sort_values(["n_configs", "runs", "n_tasks"])
print("kept=", hprint.perc(len(df_budget), len(df_curse)))
display(df_budget[["n_configs", "runs", "n_tasks", "n_seeds", "curse"]])
# Outcome: 20 of the 64 settings share their K and runs per config with
# another setting, and within each group the curse goes down as N goes up.

# %% [markdown]
# # Part 3: Picking the right config

# %% [markdown]
# ## Cell 3.1: Probability that the winner is truly best
#
# **Goal**
# - See how often the observed winner is the config with the highest $\bar{Y}_A$

# %% [markdown]
# **Description**
# - Inputs
#   - `df_curse`: the full curse sweep
# - Panels
#   - One panel per S, with the probability against N, one line per K

# %%
# Plot the probability that the winner is the truly best config.
utils.plot_prob_best(df_curse, setup_txt, figsize=(13, 3.8))

# %% [markdown]
# **Guided usage**
# - Look at K = 10, N = 100, S = 3
#   - Observe that the winner is truly best 45.5% of the time
# - Look at the largest setting, K = 20, N = 500, S = 10
#   - Observe that it is still right only 69.5% of the time

# %% [markdown]
# **Implementation** `plot_prob_best(df, setup_txt, figsize=None)`
# - Draws one panel per S, one line per K, via `_plot_lines_by()`
# - Fixes the y axis to $[0, 1]$
# - Puts the set-up of the sweep under the title

# %%
hintros.print_obj_info(utils.plot_prob_best)

# %% [markdown]
# ## Cell 3.2: Cost of a wrong pick
#
# - When the winner is wrong, the loss is small, since the configs are close
# - Rankings below the top are noisy

# %%
# Show regret and rank error at two settings, in points and places.
idx = ["n_configs", "n_tasks", "n_seeds"]
df_pick = df_curse.set_index(idx).loc[[(10, 100, 3), (20, 25, 1)]]
df_pick = df_pick.assign(regret_points=100 * df_pick["regret"])
display(df_pick[["is_truly_best", "regret_points", "rank_error"]])
# Outcome:
# - At K = 10, N = 100, S = 3 the winner's true rate is 1.1 points below the
#   best on average
# - At K = 20, N = 25, S = 1 each config's rank is off by 5.4 places on average

# %% [markdown]
# # Part 4: Fixing the curse

# %% [markdown]
# ## Cell 4.1: Fresh tasks
#
# - Rerunning the winner on N fresh tasks with S seeds gives an unbiased estimate
#   of its true rate
# - The cost is N S extra runs, i.e., 1 / K of the original search

# %%
# Pool the fresh-task error over all 64 settings.
fresh_task_mean = df_curse["fresh_task_error"].mean()
# 2.58 is the two-sided 99% critical value of a standard normal.
z_crit = 2.58
z = df_curse["fresh_task_error"] / df_curse["fresh_task_error_se"]
n_beyond = int((z.abs() > z_crit).sum())
print("fresh_task_mean=", fresh_task_mean)
print("n_beyond=", n_beyond)
# Outcome: the mean error is about -0.00002, and 1 of 64 settings is beyond
# 2.58 standard errors from zero, about what chance gives.

# %% [markdown]
# ## Cell 4.2: Fresh seeds on the same tasks
#
# - Rerunning the winner on the same tasks with new seeds removes only part of
#   the curse
# - The winner was also selected for its luck on these tasks (the $u_{ki}$
#   terms), and new seeds keep that luck
# - With more seeds, more of the curse comes from task luck and less from seed
#   luck

# %%
# Compute the share of the curse that fresh seeds remove at K = 10.
removed = 1 - df_k10["fresh_seed_error"] / df_k10["curse"]
df_removed = df_k10.assign(removed=removed).pivot(
    index="n_tasks", columns="n_seeds", values="removed"
)
display(df_removed)
# Outcome: fresh seeds remove 85% to 92% of the curse at S = 1, but only 27%
# to 38% at S = 10.

# %% [markdown]
# # Part 5: Error bars

# %% [markdown]
# ## Cell 5.1: Coverage of bootstrap intervals
#
# **Goal**
# - Check how often a 95% bootstrap interval on one config's score contains its
#   true rate $\bar{Y}_A$

# %% [markdown]
# **Description**
# - Inputs
#   - `df_cov`: the full coverage sweep, at K = 5
# - Panels
#   - `Resample tasks`: coverage against N, one line per S
#   - `Resample tasks, then seeds`: the same for the two-stage scheme
#   - The dashed line is the nominal 95%

# %%
# Describe the set-up of the coverage sweep.
setup_cov_txt = utils.describe_sweep_setup(base, n_coverage_reps)
setup_cov_txt += f", K = {coverage_k}, {n_boot} bootstrap draws per interval"
print("setup_cov_txt=", setup_cov_txt)
# Outcome: the model parameters, the replications, K, and the bootstrap draws.

# %%
# Plot the coverage of both resampling schemes.
utils.plot_coverage(df_cov, setup_cov_txt, figsize=(11, 4))

# %% [markdown]
# **Guided usage**
# - Look at `Resample tasks`
#   - Observe the lines below 95% for small N, and close to 95% at N = 500
# - Look at `Resample tasks, then seeds`
#   - Observe the lines mostly above the task-only lines, since the intervals
#     are wider

# %% [markdown]
# **Implementation** `plot_coverage(df, setup_txt, figsize=None)`
# - Draws one panel per resampling scheme, one line per S, via
#   `_plot_lines_by()`
# - Draws the nominal 95% as a dashed line
# - Does not force the y axis to $[0, 1]$, since coverage stays between 0.9 and
#   1 and the full range would squash the lines
# - Puts the set-up of the sweep under the title

# %%
hintros.print_obj_info(utils.plot_coverage)

# %% [markdown]
# ## Cell 5.2: Coverage in numbers

# %%
# Average coverage of each scheme for small and large N.
n_tasks_large = 500
df_cov_n = df_cov.assign(large_n=df_cov["n_tasks"] == n_tasks_large)
df_cov_n = df_cov_n.groupby(["method", "large_n"])["coverage"]
display(df_cov_n.agg(["mean", "min", "max"]))
# Outcome:
# - Resampling tasks covers 93.5% on average for N from 25 to 100 and 94.8% at
#   N = 500
# - Resampling tasks then seeds covers 94.8% and 95.8%, from 92.6% to 96.75%

# %% [markdown]
# - Resampling tasks undercovers by about 1.5 points for small N
#   - That matches the known small-sample bias of the percentile bootstrap, but
#     the cause was not checked here
# - The two-stage scheme is not the right method just because it is closer to
#   95%
#   - With S = 1 it is the same as task resampling
#   - With iid outcomes it overcovers (98% in the tests) because it counts seed
#     noise twice
#   - In this model the extra width happens to cancel the small-N undercoverage
# - The coverage standard errors treat the K intervals of one replication as
#   independent
#   - They share the same tasks, so the real standard errors are a bit larger

# %% [markdown]
# # Part 6: Limits of the model

# %% [markdown]
# ## Cell 6.1: Assumptions that could make a real benchmark behave differently
#
# - Task difficulty is Normal on the logit scale
#   - Real per-task results are often bimodal, with many tasks solved by every
#     config or by none
# - $u_{ki}$ is independent across tasks
#   - Real agents fail in correlated ways, e.g., one scaffold breaks on every
#     task from the same repository
# - Seeds are independent Bernoulli draws
#   - Real reruns can share cached state, rate limits, or flaky test
#     environments
# - Configs are drawn independently with one skill spread
#   - A real scaffold search produces variants of one agent whose interactions
#     $u_{ki}$ are correlated, which shrinks the task-luck part of the curse
# - The selection rule is a plain argmax over all K at once
#   - Real searches are often sequential and adaptive, which can make the curse
#     larger
# - Real benchmarks have a fixed task set, e.g., 500 tasks for SWE-bench
#   Verified
#   - Fresh tasks may not exist, and then fresh seeds are the only fix
#     available, with the limits shown in Part 4
# - All parameter values are guesses
#   - The next step is to fit `sigma_theta`, `sigma_b`, and `sigma_u` to real
#     per-task, per-seed results
