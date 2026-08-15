# Contributing

Thank you for looking at this.

This repository accompanies a paper. Its purpose is to let others reproduce and
build on the results, so at this time we do not plan to accept non-trivial
feature contributions. Keeping the released code faithful to what the paper
reports matters more here than growing it. You are of course free to fork it for
your own purposes, as the [MIT licence](LICENSE) permits.

Contributions that are very welcome:

- **Bug reports.** Especially anything where the code disagrees with the paper.
  Please include the command you ran, the full traceback, and your Python and
  package versions.
- **Reproduction reports.** If you ran an experiment and got materially
  different numbers, we would like to know. Note that every configuration in the
  paper is a single run with no error bars, so small differences are expected.
- **Fixes for genuine defects:** crashes, incorrect maths, documentation that
  describes behaviour the code does not have.

## Before opening a pull request

Run the same checks CI runs:

```sh
bash test.sh
```

That lints, builds the wheel, installs it into a scratch environment, and runs
the test suite against the installed package rather than the working tree.

## One thing to know before you edit

`moevo/evolve/seeds/*.py` are **experimental inputs, not library code.**
`run_evolve.py` reads them as text and hands them to the mutator as generation
zero. They are byte-identical to the programs used in the paper and are excluded
from the linter and the formatter so they stay that way. Reformatting them would
silently change the starting point of every reported run. Please do not tidy
them.
