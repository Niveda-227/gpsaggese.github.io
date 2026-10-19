<!-- toc -->

- [Summary](#summary)
- [Rules and Docs That Were Followed](#rules-and-docs-that-were-followed)
- [Files](#files)
- [Merging `skill_luck_sim/`](#merging-skill_luck_sim)
- [Coding Skills](#coding-skills)
- [Notebook Skills](#notebook-skills)
- [Testing Skills](#testing-skills)
- [Docs From Issue 683](#docs-from-issue-683)
- [Where a Rule Was Not Followed, and Why](#where-a-rule-was-not-followed-and-why)
- [How It Was Checked](#how-it-was-checked)
- [Last Review](#last-review)

<!-- tocstop -->

# How the `.claude` Skills Were Used

## Summary

- This file records which rules and skills from `.claude/skills` were applied
  to the agent benchmark code and notebooks in this folder, and how
- It also lists where a rule was not followed and why
- It covers the agent benchmark files (PR 1); the `causal_success_*` files will
  be added in a later PR

## Rules and Docs That Were Followed

- The repo's `.claude` points to `helpers_root/.claude`, which on this branch is
  helpers commit `13e45d2`
- The current helpers `master` has newer versions of the same rules, and they
  differ in places, e.g.:
  - The old notebook template has an `## Imports and Setup` header and calls
    `hdbg.init_logger()`
  - The new one says not to add a setup header, and wires logging through
    `utils.init_loggers(_LOG)`
- I followed the newer version, since that is the current standard
  - Every helper it calls (`hnotebook.config_notebook()`,
    `hnotebook.init_loggers()`, `hintrospection.print_obj_info()`,
    `hprint.to_str()`) exists at `13e45d2`, so nothing breaks on this branch
- I also checked the code against the docs listed in issue 683 (see
  [Docs From Issue 683](#docs-from-issue-683))

## Files

| File                                    | What It Is                                         |
| :-------------------------------------- | :------------------------------------------------- |
| `agent_benchmark_utils.py`              | The simulation, merged from `skill_luck_sim/`      |
| `agent_benchmark_API.py` / `.ipynb`     | API notebook: one small benchmark, step by step    |
| `agent_benchmark_example.py` / `.ipynb` | Example notebook: the full sweep, figures, limits  |
| `agent_benchmark_results/`              | The two CSV files of the full sweep (seed 0)       |
| `test/test_agent_benchmark_utils.py`    | Unit tests for `agent_benchmark_utils.py`          |

## Merging `skill_luck_sim/`

- The 5 files `generate.py`, `estimate.py`, `bootstrap.py`, `plotting.py`, and
  `run_sweep.py` became one file, `agent_benchmark_utils.py`
  - Its sections go from low level to high level: model parameters, model
    draws, world, scores and ranks, re-evaluation, replications, bootstrap,
    sweeps, plotting
- `run_sweep.py` was a script that also saved 3 PNG files
  - The sweeps are now functions (`run_curse_sweep()`, `run_coverage_sweep()`,
    `run_and_save_sweeps()`) called from the example notebook
  - The plots are drawn in the notebook instead of being saved, so the PNG
    files were dropped and the CSV files kept
- The results did not change
  - I ran the old and the new code with the same seeds and compared every
    output with `pandas` and `numpy` equality checks: worlds, tables,
    replications, bootstrap replicates, intervals, coverage, and both sweeps on
    a small grid
  - All outputs are identical, and the new code reproduces the first row of the
    saved `curse_sweep.csv` exactly
  - The only change in behavior is on invalid input, which now raises
    `AssertionError` instead of `ValueError` (see `coding.fix_dasserts` below)
  - The plot functions now take a `setup_txt` argument, so that each figure
    shows the set-up of the sweep under its title
- The old `README.md` had the results and the limits of the model
  - They are now markdown in the example notebook, and I re-checked every
    number against the CSV files
  - One number in the old README was wrong: the fresh-task error pooled over
    the 64 settings is -0.00002, not +0.0002, and the notebook has the correct
    value

## Coding Skills

- `coding.rules.md` and `coding.template.py`: Applied to
  `agent_benchmark_utils.py` and the test file
- `coding.fix_type_hints`: Type hints use `typing` (`List`, `Dict`, `Tuple`,
  `Optional`) instead of `list[...]` and `X | None`
- `coding.fix_dasserts`: The checks that raised `ValueError` now use
  `hdbg.dassert_lte()` with the message passed as `msg=`
  - The unknown bootstrap method is caught by the `else` branch of a complete
    `if`-`elif`-`else` with `hdbg.dfatal()`
  - The checks are the same, but invalid input now raises `AssertionError`
- `coding.fix_docstring`: Every function has a REST docstring
  - The first line is a single line
  - `:param:` and `:return:` describe what a value is, not its type
  - Examples use `E.g.,`
- `coding.fix_empty_lines`: No empty lines inside functions; steps are
  separated by comments ending with a period
- `coding.fix_comments`: Comments are full sentences that start with a capital
  letter and end with a period
- `coding.fix_param_use`: Required arguments are passed by position and
  optional ones by keyword, and every default parameter is keyword-only
  - Values are assigned to variables before the call instead of being passed
    as literals
- `coding.fix_constants`: The sweep grids, which were module constants in
  `run_sweep.py`, are now keyword parameters of the sweep functions with the
  same defaults
- `coding.fix_from_imports`: Imports are `import x.y as z`
  - The only `from` imports are `typing` and the `IPython.display` fallback of
    the notebook template
- `coding.fix_use_helpers`: `hdbg` for checks, `hio.create_dir()` for the
  output directory, `hprint.to_str()` for debug logs, `hprint.perc()` for
  filtered rows, `hnotebook.init_loggers()` for logging
- `coding.make_function_private`: Functions used only inside the file start
  with `_` (`_compute_percentile_ci()`, `_setting_seed()`, `_style_axis()`,
  `_plot_lines_by()`); the rest are called from the notebooks or the tests
- `coding.reorg_functions`: Sections go from low level to high level, with each
  function after the ones it calls
- `coding.use_single_return`: Every function has one `return`
- `coding.rules.md`, logging: Every function logs its main arguments at entry
  with `_LOG.debug()`, and functions returning a small value log it with
  `return=%s`
- `coding.rules.md`, progress bar: The sweep loops use `tqdm`

## Notebook Skills

- `notebook.rules.md` and `notebook.template.py`: Both notebooks follow the
  template
  - Title cell with the scope and "The pedagogical arc" bullets
  - An autoreload and imports cell, then the helpers, `utils`, and logging cell
  - `# Part N:` and `## Cell N.M:` headers, numbered without gaps
- Shared-utils exception of `notebook.rules.md`: Both notebooks are views of
  the same simulation, so they share one `agent_benchmark_utils.py`, named
  after the shared topic and imported as `utils`
- `notebook.create_api_intro`: `agent_benchmark_API` has the
  `## Library Overview` section and ends with `## Summary: The Mental Model`
  - The overview has the problem, the key abstraction, a mental model table,
    and the key classes
- `notebook.split_cells`: One task per code cell, each ending with an
  `# Outcome:` comment that I checked against the actual output
- `notebook.format_md_cells`: Markdown is wrapped at 85 characters, uses nested
  bullets and sentence case, and is ASCII only
- `notebook.format_interactive_cells`: Each plot follows Goal, Description
  (Inputs and Panels), the plot, Guided usage, Implementation, and
  `hintros.print_obj_info()`
- `notebook.fix_logging`: The notebooks use the `init_loggers()` idiom of the
  template
- `notebook.refactor_to_utils`: All plotting and simulation code is in the
  utils file, and the cells only call it
- `notebook.lint_numbered_cells`: Cell numbers are sequential (see the
  deviation below on function names)

## Testing Skills

- `testing.rules.md` and `testing.template.py`: Applied to
  `test/test_agent_benchmark_utils.py`
  - Tests use `hunitest.TestCase`, classes are named `Test_<function>`, and
    methods `test1`, `test2`, ...
  - Each test has a docstring and the sections `# Prepare inputs.`,
    `# Prepare outputs.`, `# Run test.`, `# Check outputs.`
  - Values are assigned to variables before calling the function under test
  - Each test method checks a single case, and shared code is in a `helper()`
    method placed first, not in `setUp()`
  - `assert_equal()` with `str()` for exact structures, `assertEqual()` and
    `assertLessEqual()` for simple values, `numpy` checks for floats with a
    tolerance
  - The 3 slow tests are marked `@pytest.mark.slow`
- The 20 tests of `skill_luck_sim/test/` were ported, with 3 changes
  - Dropped: The test that an unknown bootstrap method raises, because the
    rules say not to test error conditions heavily
  - Split: The test of constant outcomes looped over both bootstrap methods,
    and is now 2 tests, one per method
  - New: `Test_run_curse_sweep.test1`, which checks that a sweep setting gives
    the same row alone and inside a larger grid
- `Test_winners_curse` and `Test_reevaluation` were renamed to
  `Test_run_replications` and `Test_summarize_replications`, to follow the
  `Test_<function>` naming

## Docs From Issue 683

- The code was also checked against the docs that issue 683 lists
  - `all.coding_style.how_to_guide.md`
  - `all.write_unit_tests.how_to_guide.md` and
    `all.run_unit_tests.how_to_guide.md`
  - `all.jupyter_notebook.how_to_guide.md`
  - `all.writing_docs.how_to_guide.md`
  - `all.submit_code_for_review.how_to_guide.md` and
    `all.code_review.how_to_guide.md`
- Changes made because of these docs
  - Each plot has a title and the set-up of the sweep (model parameters,
    replications, bootstrap draws), from _Make each plot self-explanatory_
  - Each filter in the example notebook prints how many rows it kept with
    `hprint.perc()`, from _Always look at the discarded data_
  - This file has a table of contents, a summary, and a last review section,
    from _Format of each doc_
- Where these docs and the newer `.claude` rules disagree, I followed the
  `.claude` rules
  - Notebooks import their utils as `utils` and show results with `print()` and
    `display()`, as the notebook template does
  - Test sections are `# Run test.` and `# Check outputs.`, as in
    `testing.rules.md`
  - Notebook headings use sentence case, as in `notebook.format_md_cells`
  - `:param figsize:` says what its default is, as in `coding.template.py`
- Not changed
  - Column names like `"n_tasks"` are not module constants: they are used by
    many functions and the notebooks, and the change would add noise to the
    diff
  - The coverage plot does not force its y axis to [0, 1], since coverage stays
    between 0.9 and 1 and the full range would squash the lines

## Where a Rule Was Not Followed, and Why

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
- "Guided usage" is written for static plots ("Look at ... Observe ..."),
  since the notebooks have no widgets
- `tqdm` is called with `ascii=True`, because the default bar uses non-ASCII
  block characters, which the notebook rules do not allow in output
- `ruff` 0.16 with this repo's `pyproject.toml` (`fix = true`) rewrites
  `typing` hints into `list[...]` and `import helpers.hdbg as hdbg` into
  `from helpers import hdbg`, which contradicts `coding.rules.md`
  - I followed `coding.rules.md` and ran `ruff` with `--no-fix`
- I could not run `i lint`, since the cloud environment used for this PR does
  not have the `invoke` setup
  - Formatting was checked with `ruff format --check` instead

## How It Was Checked

- Tests: All 21 pass, and 18 pass when skipping the slow ones with
  `-m "not slow"`
  ```bash
  > python -m pytest \
      research/Causal_Analysis_of_Agent_Skill_And_Luck/test/test_agent_benchmark_utils.py
  ```
- Notebooks: Both were run top to bottom with `jupyter nbconvert --execute`,
  and the saved outputs have no errors and no non-ASCII characters
- Formatting: `ruff format --check` passes on all Python files

## Last Review

- Niveda on 2026-10-19
