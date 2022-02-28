# Neural: a simple python package for neuron and synapse models

This pinned version of the neural package accompanies the other [repo](https://github.com/TK-21st/AntennalLobeLLY22).

## Installation
Python `>=3.8` is recommended.

One in-house dependency is required: [`pycodgen`](https://github.com/chungheng/pycodegen) which can be installed as:
```bash
git clone git@github.com:chungheng/pycodegen.git
cd pycodegen
pip install .
```

After installing `pycodegen`, install `neural` as follows:
```
cd ..  # assuming previously in `pycodegen/`
git clone --single-branch --branch al git@github.com:chungheng/neural.git
cd neural
pip install .
```
