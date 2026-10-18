# How the `.claude` Skills Were Used

This file records which rules and skills from `.claude/skills` were applied to the
code and notebooks in this folder, how, and where I did not follow a rule and why.
It covers the agent benchmark files (PR 1). The `causal_success_*` files will be
added in a later PR

## Which Version of the Rules

- The repo's `.claude` points to `helpers_root/.claude`, which on this branch is
  helpers commit `13e45d2` (2026-08-16)
- The current helpers `master` (commit `58ff0bc`) has newer versions of the same
  rules, and they differ in places, e.g.:
  - The old notebook template has an `## Imports and Setup` header and calls
    `hdbg.init_logger()`
  - The new one says not to add a setup header, and wires logging through
    `utils.init_loggers(_LOG)`
- I followed the newer version, since that is the current standard. I checked
  that every helper it calls (`hnotebook.config_notebook()`,
  `hnotebook.init_loggers()`, `hintrospection.print_obj_info()`,
  `hprint.to_str()`) exists at `13e45d2`, so nothing breaks on this branch

## Files

| File | What it is |
| ---- | ---------- |
| `agent_benchmark_utils.py` | The simulation, merged from the 4 modules of `skill_luck_sim/` |
| `agent_benchmark_API.py` / `.ipynb` | API notebook: one small benchmark, step by step |
| `agent_benchmark_example.py` / `.ipynb` | Example notebook: the full sweep, figures, limits |
| `agent_benchmark_results/` | The two CSV files of the full sweep (seed 0) |
| `test/test_agent_benchmark_utils.py` | Unit tests for `agent_benchmark_utils.py` |

## Merging `skill_luck_sim/`

- `generate.py`, `estimate.py`, `bootstrap.py`, `plotting.py`, and `run_sweep.py`
  became one file, `agent_benchmark_utils.py`, with sections ordered from low
  level to high level:
  - Model parameters, model draws, world, scores and ranks, re-evaluation,
    replications, bootstrap, sweeps, plotting
- `run_sweep.py` was a script that also saved 3 PNG files
  - The sweeps are now functions (`run_curse_sweep()`, `run_coverage_sweep()`,
    `run_and_save_sweeps()`) called from the example notebook
  - The plots are drawn in the notebook instead of being saved, so the PNG files
    were dropped and the CSV files kept
- The behavior did not change, except for the exception type on invalid input
  (see `coding.fix_dasserts` below)
  - I ran the old and the new code with the same seeds and compared every output
    with `pandas` and `numpy` equality checks: worlds, tables, replications,
    bootstrap replicates, intervals, coverage, and both sweeps on a small grid
  - All outputs are identical, and the new code reproduces the first row of the
    saved `curse_sweep.csv` exactly
- The old `README.md` had the results and the limits of the model. They are now
  markdown in the example notebook, and I re-checked every number against the CSV
  files
  - One number in the old README was wrong: the fresh-task error pooled over the
    64 settings is -0.00002, not +0.0002. The notebook has the correct value

## Coding Skills

- `coding.rules.md` and `coding.template.py`: applied to
  `agent_benchmark_utils.py` and the test file
- `coding.fix_type_hints`: type hints use `typing` (`List`, `Dict`, `Tuple`,
  `Optional`) instead of `list[...]` and `X | None`
- `coding.fix_dasserts`: the checks that raised `ValueError` now use
  `hdbg.dassert_lte()` and `hdbg.dassert_in()`, each with a message
  - The checks are the same, but invalid input now raises `AssertionError`
    instead of `ValueError`. This is the only change in behavior, and it only
    affects invalid input
- `coding.fix_docstring`: every function has a REST docstring with `:param:`
  and, when it returns a value, `:return:`, and examples use `E.g.,`
- `coding.fix_empty_lines`: no empty lines inside functions; steps are separated
  by comments ending with a period
- `coding.fix_comments`: comments are full sentences ending with a period
- `coding.fix_param_use`: required arguments are passed by position and optional
  ones by keyword, and every default parameter is keyword-only after `*`
- `coding.fix_constants`: the sweep grids, which were module constants in
  `run_sweep.py`, are now keyword parameters of the sweep functions with the same
  defaults
- `coding.fix_from_imports`: imports are `import x.y as z`
  - The only `from` imports are `typing` and the `IPython.display` fallback of
    the notebook template
- `coding.fix_use_helpers`: `hdbg` for checks, `hio.create_dir()` for the output
  directory, `hprint.to_str()` for debug logs, `hnotebook.init_loggers()` for
  logging
- `coding.make_function_private`: functions used only inside the file start with
  `_` (`_compute_percentile_ci()`, `_setting_seed()`, `_style_axis()`,
  `_plot_lines_by()`); the rest are called from the notebooks or the tests
- `coding.reorg_functions`: sections go from low level to high level, with each
  function after the ones it calls
- `coding.use_single_return`: every function has one `return`
- `coding.rules.md`, logging: every function logs its main arguments at entry
  with `_LOG.debug()`, and functions returning a small value log it with
  `return=%s`
- `coding.rules.md`, progress bar: the sweep loops use `tqdm`

## Notebook Skills

- `notebook.rules.md` and `notebook.template.py`: both notebooks follow the
  template
  - Title cell with the scope and "The pedagogical arc" bullets
  - An autoreload and imports cell, then the helpers, `utils`, and logging cell
  - `# Part N:` and `## Cell N.M:` headers, numbered without gaps
- Shared-utils exception of `notebook.rules.md`: both notebooks are views of the
  same simulation, so they share one `agent_benchmark_utils.py`, named after the
  shared topic and imported as `utils`
- `notebook.create_api_intro`: `agent_benchmark_API` has the
  `## Library Overview` section, with the problem, the key abstraction, a mental
  model table, and the key classes, and ends with `## Summary: The Mental Model`
- `notebook.split_cells`: one task per code cell, each ending with an
  `# Outcome:` comment that I checked against the actual output
- `notebook.format_md_cells`: markdown is wrapped at 85 characters, uses nested
  bullets and sentence case, and is ASCII only
- `notebook.format_interactive_cells`: each plot follows Goal, Description
  (Inputs and Panels), the plot, Guided usage, Implementation, and
  `hintros.print_obj_info()`
- `notebook.fix_logging`: the notebooks use the `init_loggers()` idiom of the
  template
- `notebook.refactor_to_utils`: all plotting and simulation code is in the utils
  file, and the cells only call it
- `notebook.lint_numbered_cells`: cell numbers are sequential (see the deviation
  below on function names)

## Testing Skills

- `testing.rules.md` and `testing.template.py`: applied to
  `test/test_agent_benchmark_utils.py`
  - Tests use `hunitest.TestCase`, classes are named `Test_<function>`, methods
    `test1`, `test2`, ...
  - Each test has a docstring and the sections `# Prepare inputs.`,
    `# Prepare outputs.`, `# Run test.`, `# Check outputs.`
  - Shared code is in a `helper()` method placed first, not in `setUp()`
  - `assert_equal()` with `str()` for exact structures, `assertEqual()` for simple
    values, `numpy` checks for floats with a tolerance
  - The 3 slow tests are marked `@pytest.mark.slow`
- All 20 tests of `skill_luck_sim/test/` were ported except one, plus one new test
  - Dropped: the test that an unknown bootstrap method raises, because the rules
    say not to test error conditions heavily
  - New: `Test_run_curse_sweep.test1`, which checks that a sweep setting gives the
    same row alone and inside a larger grid
  - `Test_winners_curse` and `Test_reevaluation` were renamed to
    `Test_run_replications` and `Test_summarize_replications`, to follow the
    `Test_<function>` naming

## Where I Did Not Follow a Rule, and Why

- Function names are not numbered by cell (e.g., `plot_curse()` instead of
  `cell2_1_plot_curse()`)
  - These functions are the API of the simulation: the API notebook teaches
    them, and the tests and both notebooks call them
  - Renaming them after cells of one notebook would tie a shared library to one
    notebook's layout
- The plots use `matplotlib` `errorbar()` instead of `seaborn`
  - The error bars come from Monte Carlo standard errors that are already
    computed in the sweep tables, and `seaborn` cannot draw precomputed error
    bars directly
- "Guided usage" is written for static plots ("Look at ... Observe ..."), since
  the notebooks have no widgets
- `tqdm` is called with `ascii=True`, because the default bar uses non-ASCII
  block characters, which the notebook rules do not allow in output
- `ruff` 0.16 with this repo's `pyproject.toml` (`fix = true`) rewrites `typing`
  hints into `list[...]` and `import helpers.hdbg as hdbg` into
  `from helpers import hdbg`, which contradicts `coding.rules.md`
  - I followed `coding.rules.md` and ran `ruff` with `--no-fix`

## How It Was Checked

- Tests: all 20 pass (17 when skipping the slow ones with `-m "not slow"`)
  ```bash
  > python -m pytest \
      research/Causal_Analysis_of_Agent_Skill_And_Luck/test/test_agent_benchmark_utils.py
  ```
- Notebooks: both were run top to bottom with `jupyter nbconvert --execute`, and
  the saved outputs have no errors and no non-ASCII characters
- Formatting: `ruff format --check` passes on all Python files
