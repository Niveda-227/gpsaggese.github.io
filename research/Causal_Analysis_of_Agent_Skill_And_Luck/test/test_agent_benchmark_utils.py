"""
Import as:

import research.Causal_Analysis_of_Agent_Skill_And_Luck.test.test_agent_benchmark_utils as rcaoasaltt
"""

import logging
import math
from typing import Dict

import numpy as np
import pandas as pd
import pytest

import helpers.hunit_test as hunitest
import research.Causal_Analysis_of_Agent_Skill_And_Luck.agent_benchmark_utils as rcaoasalabu

_LOG = logging.getLogger(__name__)


def _expected_max_binomial_mean(n_configs: int, n_trials: int) -> float:
    """
    Compute `E[max_k X_k / n]` for `n_configs` iid `Binomial(n, 0.5)`.

    Use `E[max] = sum_x x (F(x)^K - F(x - 1)^K)`, with `F` the binomial CDF.

    :param n_configs: number of iid variables K
    :param n_trials: number of trials n of each variable
    :return: expected maximum, divided by `n_trials`
    """
    pmf = np.array([math.comb(n_trials, x) for x in range(n_trials + 1)]) / (
        2.0**n_trials
    )
    cdf = np.cumsum(pmf)
    cdf_prev = np.concatenate([[0.0], cdf[:-1]])
    x = np.arange(n_trials + 1)
    expected = (
        float((x * (cdf**n_configs - cdf_prev**n_configs)).sum()) / n_trials
    )
    return expected


# #############################################################################
# Test_compute_true_resolve_rate
# #############################################################################


class Test_compute_true_resolve_rate(hunitest.TestCase):
    """
    Test the numerical integration of `Ybar_A` over the task population.
    """

    def test1(self) -> None:
        """
        Test that `theta = mu_b` gives exactly 0.5, by symmetry of the logit.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(mu_b=0.7, sigma_b=2.0, sigma_u=1.0)
        theta = np.array([0.7])
        # Prepare outputs.
        expected = [0.5]
        # Run test.
        ybar = rcaoasalabu.compute_true_resolve_rate(theta, params)
        # Check outputs.
        np.testing.assert_allclose(ybar, expected, atol=1e-12)

    def test2(self) -> None:
        """
        Test that with no task spread `Ybar_A` is `sigmoid(theta - mu_b)`.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(mu_b=0.5, sigma_b=0.0, sigma_u=0.0)
        theta = np.array([-1.0, 0.5, 2.0])
        # Prepare outputs.
        expected = 1.0 / (1.0 + np.exp(-(theta - 0.5)))
        # Run test.
        ybar = rcaoasalabu.compute_true_resolve_rate(theta, params)
        # Check outputs.
        np.testing.assert_allclose(ybar, expected, atol=1e-12)

    def test3(self) -> None:
        """
        Test that the integral matches a large Monte Carlo average.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(sigma_b=2.0, sigma_u=1.0)
        theta = np.array([1.3])
        seed = 0
        n_draws = 2_000_000
        # Prepare outputs.
        rng = np.random.default_rng(seed)
        b = rng.normal(params.mu_b, params.sigma_b, n_draws)
        u = rng.normal(0.0, params.sigma_u, n_draws)
        ybar_mc = rcaoasalabu.sigmoid(theta[0] - b + u).mean()
        # Run test.
        ybar = rcaoasalabu.compute_true_resolve_rate(theta, params)[0]
        # Check outputs. The Monte Carlo standard error is about 0.0002.
        self.assertAlmostEqual(ybar, ybar_mc, delta=0.001)


# #############################################################################
# Test_draw_world
# #############################################################################


class Test_draw_world(hunitest.TestCase):
    """
    Test shapes and reproducibility of a drawn world.
    """

    def test1(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        seed = 11
        # Run test.
        world1 = rcaoasalabu.draw_world(params, np.random.default_rng(seed))
        world2 = rcaoasalabu.draw_world(params, np.random.default_rng(seed))
        # Check outputs.
        for field in ["theta", "ybar", "b", "u", "y"]:
            np.testing.assert_array_equal(
                getattr(world1, field), getattr(world2, field)
            )

    def test2(self) -> None:
        """
        Test that different seeds give different outcomes.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        seed1 = 1
        seed2 = 2
        # Run test.
        world1 = rcaoasalabu.draw_world(params, np.random.default_rng(seed1))
        world2 = rcaoasalabu.draw_world(params, np.random.default_rng(seed2))
        # Check outputs.
        self.assertFalse(np.array_equal(world1.y, world2.y))

    def test3(self) -> None:
        """
        Test array shapes and that outcomes are 0 or 1.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(n_configs=4, n_tasks=30, n_seeds=3)
        seed = 0
        # Run test.
        world = rcaoasalabu.draw_world(params, np.random.default_rng(seed))
        # Check outputs.
        self.assertEqual(world.theta.shape, (4,))
        self.assertEqual(world.ybar.shape, (4,))
        self.assertEqual(world.b.shape, (30,))
        self.assertEqual(world.u.shape, (4, 30))
        self.assertEqual(world.y.shape, (4, 30, 3))
        self.assertLessEqual(set(np.unique(world.y)), {0, 1})


# #############################################################################
# Test_outcomes_to_frame
# #############################################################################


class Test_outcomes_to_frame(hunitest.TestCase):
    """
    Test conversion of the outcome array to the generic long table.
    """

    def test1(self) -> None:
        """
        Test columns and that each row matches the array entry.
        """
        # Prepare inputs.
        seed = 0
        y = np.random.default_rng(seed).integers(0, 2, size=(2, 3, 4))
        # Prepare outputs.
        expected_columns = "['config', 'task', 'seed', 'resolved']"
        # Run test.
        df = rcaoasalabu.outcomes_to_frame(y)
        # Check outputs.
        self.assert_equal(str(list(df.columns)), expected_columns)
        self.assertEqual(len(df), 24)
        actual = df["resolved"].to_numpy()
        expected = y[df["config"], df["task"], df["seed"]]
        np.testing.assert_array_equal(actual, expected)


# #############################################################################
# Test_compute_ranks
# #############################################################################


class Test_compute_ranks(hunitest.TestCase):
    """
    Test ranking from best to worst with ties.
    """

    def test1(self) -> None:
        """
        Test the docstring example with a tie.
        """
        # Prepare inputs.
        values = np.array([0.2, 0.5, 0.5])
        # Prepare outputs.
        expected = "[3.0, 1.5, 1.5]"
        # Run test.
        actual = rcaoasalabu.compute_ranks(values)
        # Check outputs.
        self.assert_equal(str(actual.tolist()), expected)


# #############################################################################
# Test_run_replications
# #############################################################################


class Test_run_replications(hunitest.TestCase):
    """
    Test the size of the winner's curse and the reproducibility of a batch.
    """

    @pytest.mark.slow
    def test1(self) -> None:
        """
        Test that with all configs equal the curse matches order statistics.

        With `sigma_theta = sigma_b = sigma_u = 0` and `theta = mu_b`, every
        outcome is Bernoulli(0.5), every `Ybar_A` is 0.5, and each score is
        `Binomial(N S, 0.5) / (N S)`, independent across configs. So the
        expected curse is `E[max of K such scores] - 0.5`.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(
            n_configs=5,
            n_tasks=50,
            n_seeds=2,
            mu_theta=0.0,
            sigma_theta=0.0,
            sigma_b=0.0,
            sigma_u=0.0,
        )
        n_reps = 4000
        seed = 0
        # Prepare outputs. For K = 5 the normal approximation gives
        # 1.163 * 0.5 / sqrt(100) = 0.058.
        n_trials = params.n_tasks * params.n_seeds
        expected = _expected_max_binomial_mean(params.n_configs, n_trials) - 0.5
        # Run test.
        df = rcaoasalabu.run_replications(params, n_reps, seed)
        summary = rcaoasalabu.summarize_replications(df, params)
        # Check outputs.
        self.assertGreater(summary["curse"], 0.0)
        self.assertAlmostEqual(
            summary["curse"], expected, delta=3 * summary["curse_se"]
        )
        self.assertTrue(np.isnan(summary["is_truly_best"]))

    def test2(self) -> None:
        """
        Test that the curse is positive when configs are equal but tasks vary.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(
            n_configs=10, n_tasks=25, n_seeds=1, sigma_theta=0.0
        )
        n_reps = 1000
        seed = 0
        # Run test.
        df = rcaoasalabu.run_replications(params, n_reps, seed)
        summary = rcaoasalabu.summarize_replications(df, params)
        # Check outputs.
        self.assertGreater(summary["curse"], 5 * summary["curse_se"])

    def test3(self) -> None:
        """
        Test that the curse goes toward zero with many tasks and seeds.
        """
        # Prepare inputs.
        small = rcaoasalabu.SimParams(n_configs=10, n_tasks=25, n_seeds=1)
        large = rcaoasalabu.SimParams(n_configs=10, n_tasks=2000, n_seeds=20)
        n_reps_small = 300
        n_reps_large = 100
        seed = 0
        # Run test.
        df_small = rcaoasalabu.run_replications(small, n_reps_small, seed)
        curse_small = rcaoasalabu.summarize_replications(df_small, small)[
            "curse"
        ]
        df_large = rcaoasalabu.run_replications(large, n_reps_large, seed)
        curse_large = rcaoasalabu.summarize_replications(df_large, large)[
            "curse"
        ]
        # Check outputs.
        self.assertLess(abs(curse_large), 0.005)
        self.assertLess(abs(curse_large), curse_small / 10)

    def test4(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(n_configs=5, n_tasks=20, n_seeds=2)
        n_reps = 20
        seed = 5
        # Run test.
        df1 = rcaoasalabu.run_replications(params, n_reps, seed)
        df2 = rcaoasalabu.run_replications(params, n_reps, seed)
        # Check outputs.
        pd.testing.assert_frame_equal(df1, df2)


# #############################################################################
# Test_summarize_replications
# #############################################################################


@pytest.mark.slow
class Test_summarize_replications(hunitest.TestCase):
    """
    Test the bias of re-evaluating the winner on fresh seeds or tasks.
    """

    def helper(self) -> Dict[str, float]:
        """
        Run a setting with a large curse: many configs, few tasks.

        :return: summary of the replications
        """
        params = rcaoasalabu.SimParams(n_configs=20, n_tasks=25, n_seeds=3)
        n_reps = 3000
        seed = 1
        df = rcaoasalabu.run_replications(params, n_reps, seed)
        summary = rcaoasalabu.summarize_replications(df, params)
        return summary

    def test1(self) -> None:
        """
        Test that fresh tasks give an unbiased estimate of the true rate.
        """
        # Run test.
        summary = self.helper()
        # Check outputs.
        self.assertGreater(summary["curse"], 10 * summary["curse_se"])
        self.assertLess(
            abs(summary["fresh_task_error"]), 3 * summary["fresh_task_error_se"]
        )

    def test2(self) -> None:
        """
        Test that fresh seeds on the same tasks remove only part of the curse.

        The winner was also selected for its luck on these tasks (through
        `u_ki`), and rerunning seeds keeps that luck.
        """
        # Run test.
        summary = self.helper()
        # Check outputs.
        self.assertGreater(
            summary["fresh_seed_error"], 5 * summary["fresh_seed_error_se"]
        )
        self.assertLess(summary["fresh_seed_error"], summary["curse"])


# #############################################################################
# Test_bootstrap_scores
# #############################################################################


class Test_bootstrap_scores(hunitest.TestCase):
    """
    Test the two resampling schemes on small inputs.
    """

    def helper(self, method: str) -> None:
        """
        Test that identical tasks with no seed variation give a constant.

        :param method: resampling scheme to test
        """
        # Prepare inputs.
        y = np.ones((10, 3), dtype=np.int8)
        n_boot = 50
        seed = 0
        rng = np.random.default_rng(seed)
        # Prepare outputs.
        expected = np.ones(n_boot)
        # Run test.
        replicates = rcaoasalabu.bootstrap_scores(y, n_boot, method, rng)
        # Check outputs.
        np.testing.assert_array_equal(replicates, expected)

    def test1(self) -> None:
        """
        Test that `task` resampling of constant outcomes gives a constant.
        """
        # Prepare inputs.
        method = "task"
        # Run test.
        self.helper(method)

    def test2(self) -> None:
        """
        Test that `task_seed` resampling of constant outcomes gives a constant.
        """
        # Prepare inputs.
        method = "task_seed"
        # Run test.
        self.helper(method)

    def test3(self) -> None:
        """
        Test that resampling tasks keeps the seeds of a task together.

        Each task is either always or never solved, so with the `task` scheme
        every replicate is a multiple of 1 / N.
        """
        # Prepare inputs.
        y = np.repeat(np.array([[1], [0], [1], [0]]), 5, axis=1)
        n_boot = 200
        method = "task"
        seed = 0
        rng = np.random.default_rng(seed)
        # Run test.
        replicates = rcaoasalabu.bootstrap_scores(y, n_boot, method, rng)
        # Check outputs.
        np.testing.assert_allclose(replicates * 4, np.round(replicates * 4))


# #############################################################################
# Test_run_coverage
# #############################################################################


class Test_run_coverage(hunitest.TestCase):
    """
    Test interval coverage in a case with a known answer.
    """

    def helper(self) -> pd.DataFrame:
        """
        Run one config whose outcomes are all iid Bernoulli(0.5).

        :return: coverage indexed by method
        """
        params = rcaoasalabu.SimParams(
            n_configs=1,
            n_tasks=100,
            n_seeds=2,
            mu_theta=0.0,
            sigma_theta=0.0,
            sigma_b=0.0,
            sigma_u=0.0,
        )
        n_reps = 600
        n_boot = 400
        seed = 0
        df = rcaoasalabu.run_coverage(params, n_reps, n_boot, seed)
        df = df.set_index("method")
        return df

    def test1(self) -> None:
        """
        Test that task resampling covers `Ybar_A = 0.5` about 95% of the time.

        The score is a mean of 200 iid Bernoulli(0.5) draws, so the
        percentile interval is close to exact. The Monte Carlo standard
        error of the coverage is about 0.009.
        """
        # Run test.
        df = self.helper()
        # Check outputs.
        self.assertAlmostEqual(df.loc["task", "coverage"], 0.95, delta=0.03)

    def test2(self) -> None:
        """
        Test that two-stage resampling overcovers, as its variance predicts.

        With S = 2, resampling seeds inside resampled tasks adds a variance of
        `(S - 1) / S * 0.25 / (N S)` on top of the task-level
        `0.25 / (N S)`, so the interval is `sqrt(1.5) = 1.22` times too wide
        and should cover `P(|Z| < 1.96 * 1.22) = 0.984` of the time.
        """
        # Run test.
        df = self.helper()
        # Check outputs.
        self.assertGreater(df.loc["task_seed", "coverage"], 0.97)

    def test3(self) -> None:
        """
        Test that the same seed gives identical output.
        """
        # Prepare inputs.
        params = rcaoasalabu.SimParams(n_configs=3, n_tasks=20, n_seeds=2)
        n_reps = 5
        n_boot = 50
        seed = 3
        # Run test.
        df1 = rcaoasalabu.run_coverage(params, n_reps, n_boot, seed)
        df2 = rcaoasalabu.run_coverage(params, n_reps, n_boot, seed)
        # Check outputs.
        pd.testing.assert_frame_equal(df1, df2)


# #############################################################################
# Test_run_curse_sweep
# #############################################################################


class Test_run_curse_sweep(hunitest.TestCase):
    """
    Test that a sweep setting does not depend on the other settings.
    """

    def test1(self) -> None:
        """
        Test that a setting gives the same row alone and inside a larger grid.
        """
        # Prepare inputs.
        base = rcaoasalabu.SimParams()
        n_reps = 10
        seed = 0
        n_tasks_grid = (25,)
        # Run test.
        df_alone = rcaoasalabu.run_curse_sweep(
            base,
            n_reps,
            seed,
            n_configs_grid=(5,),
            n_tasks_grid=n_tasks_grid,
            n_seeds_grid=(3,),
        )
        df_grid = rcaoasalabu.run_curse_sweep(
            base,
            n_reps,
            seed,
            n_configs_grid=(3, 5),
            n_tasks_grid=n_tasks_grid,
            n_seeds_grid=(1, 3),
        )
        # Check outputs.
        mask = (df_grid["n_configs"] == 5) & (df_grid["n_seeds"] == 3)
        df_row = df_grid[mask].reset_index(drop=True)
        pd.testing.assert_frame_equal(df_alone, df_row)
