# Multi-agent reinforcement learning for credit theory of money and #MMT.

[![PyPI](https://img.shields.io/pypi/v/credit-mmt-marl.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/credit-mmt-marl.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/credit-mmt-marl)][python version]
[![License](https://img.shields.io/pypi/l/credit-mmt-marl)][license]

[![Read the documentation at https://credit-mmt-marl.readthedocs.io/](https://img.shields.io/readthedocs/credit-mmt-marl/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/gatoniel/credit-mmt-marl/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/gatoniel/credit-mmt-marl/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/credit-mmt-marl/
[status]: https://pypi.org/project/credit-mmt-marl/
[python version]: https://pypi.org/project/credit-mmt-marl
[read the docs]: https://credit-mmt-marl.readthedocs.io/
[tests]: https://github.com/gatoniel/credit-mmt-marl/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/gatoniel/credit-mmt-marl
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Motivation

Germany has released its debt brake, the EU is thinking about joint debt issuance for defence spending, and the current US administration is trying to cut spending as much as possible. At the same time, Bitcoin and other crypto-currencies position themselves as real alternatives to government issued currencies. With this project, a multi-agent AI is forced to develop a monetary system from scratch. This will answer two questions:

1. What kind of monetary system will the AI come up with? A credit system or a system resembling the features of Bitcoin?
2. Which debt-surrounding rules will the AI implement?

## The multi-agent game

The AI model is set up as a multi-agent game where each player is an independent AI agent. The game can be separated into two separate parts, _production_ and _trade_, or _real_ and _financial_, respectively.

### The _real_ part - goods and production of goods

Currently, this part of the game is rather naive. It follows two-factor Cobb-Douglas functions. There are three different ressources, `energy`, `goods`, and `capital`. However, the ressources `goods` and `capital` are each separated in _self-produced_ and _bought_ to enforce trade between the players. Hence, each player $`i`$ has five ressources:

- Energy $`E_i \sim Exp(\lambda=1)`$ is randomly distributed and drawn at each time step.
- Self-produced goods $`G_{s,i}`$ are the goods the player has produced himself so far and not sold yet.
- Bought / traded goods $`G_{t,i}`$ are the goods the player has acquired from other players through trade.
- Self-produced capital $`C_{s,i}`$
- Bought / traded capital $`C_{t,i}`$

At each timestep the quantities follow these changes:

- $`\Delta G_{s,i} = {C_{s,i}}^\alpha * E_i^{1-\alpha}`$
- $`\Delta C_{s,i} = {C_{t,i}}^\beta * G_{t,i}^{1-\beta}`$

While the traded goods and capitals have to be exchanged actively by trade with other players (see below).

## Features

- TODO

## Requirements

- TODO

## Installation

You can install _Multi-agent reinforcement learning for credit theory of money and #MMT._ via [pip] from [PyPI]:

```console
$ pip install credit-mmt-marl
```

## Usage

Please see the [Command-line Reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [GPL 3.0 license][license],
_Multi-agent reinforcement learning for credit theory of money and #MMT._ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/gatoniel/credit-mmt-marl/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/gatoniel/credit-mmt-marl/blob/main/LICENSE
[contributor guide]: https://github.com/gatoniel/credit-mmt-marl/blob/main/CONTRIBUTING.md
[command-line reference]: https://credit-mmt-marl.readthedocs.io/en/latest/usage.html
