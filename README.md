# MedCoLab

This repository contains the basic training code for MedCoLab.

## Installation

```bash
pip install -e .
```

## Training

Run the default federated training experiment with:

```bash
flwr run .
```

The main training entry points are `fl/server_app.py` and `fl/client_app.py`. Basic experiment settings, such as the number of clients, training rounds, learning rate, and local epochs, can be changed in `pyproject.toml`.

The active dataset is configured in `fl/task.py` through the corresponding dataset configuration.

Training outputs are saved under `outputs/`.
