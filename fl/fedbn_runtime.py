"""Shared, persistent FedBN runtime helpers.

FedBN is a parameter-localisation policy.  It keeps BatchNorm affine
parameters and running statistics at each client while the server optimizer
aggregates all remaining trainable parameters.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from flwr.app import ArrayRecord


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bn_module_names(model: torch.nn.Module) -> set[str]:
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    }


def shared_parameter_names(model: torch.nn.Module) -> list[str]:
    """Return all trainable parameter names except BatchNorm parameters."""
    bn_names = _bn_module_names(model)
    return [
        name
        for name, _ in model.named_parameters()
        if (name.rsplit(".", 1)[0] if "." in name else "") not in bn_names
    ]


def local_state_names(model: torch.nn.Module) -> list[str]:
    """Return state not transmitted to the server under FedBN.

    This intentionally includes non-parameter buffers in addition to BN state.
    For the current detection model that includes domain_soft_targets and keeps
    the audited 348 shared / 671 local layout.
    """
    shared = set(shared_parameter_names(model))
    return [name for name in model.state_dict() if name not in shared]


def layout(model: torch.nn.Module) -> dict[str, Any]:
    shared = shared_parameter_names(model)
    local = local_state_names(model)
    return {
        "shared_count": len(shared),
        "local_count": len(local),
        "shared_keys": shared,
        "local_keys": local,
    }


def shared_array_record(model: torch.nn.Module) -> ArrayRecord:
    params = dict(model.named_parameters())
    return ArrayRecord({name: params[name].detach().cpu() for name in shared_parameter_names(model)})


def load_shared_array_record(model: torch.nn.Module, arrays: ArrayRecord) -> None:
    expected = shared_parameter_names(model)
    incoming = list(arrays.keys())
    if set(incoming) != set(expected):
        missing = sorted(set(expected) - set(incoming))
        extra = sorted(set(incoming) - set(expected))
        raise RuntimeError(
            f"FedBN shared-key mismatch: expected={len(expected)} incoming={len(incoming)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, array in arrays.items():
            target = params[name]
            value = torch.from_numpy(array.numpy()).to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise RuntimeError(f"FedBN shape mismatch for {name}: {value.shape} != {target.shape}")
            target.copy_(value)


def capture_local_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {name: state[name].detach().cpu().clone() for name in local_state_names(model)}


def load_local_state(model: torch.nn.Module, local_state: Mapping[str, torch.Tensor]) -> None:
    expected = local_state_names(model)
    incoming = list(local_state)
    if set(incoming) != set(expected):
        missing = sorted(set(expected) - set(incoming))
        extra = sorted(set(incoming) - set(expected))
        raise RuntimeError(
            f"FedBN local-key mismatch: expected={len(expected)} incoming={len(incoming)} "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    state = model.state_dict()
    with torch.no_grad():
        for name, value in local_state.items():
            target = state[name]
            value = value.to(device=target.device, dtype=target.dtype)
            if value.shape != target.shape:
                raise RuntimeError(f"FedBN local shape mismatch for {name}: {value.shape} != {target.shape}")
            target.copy_(value)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


class FedBNStateStore:
    """Durable per-experiment, per-client BN/local-state storage."""

    def __init__(self, root: str | Path, expected_clients: int = 5):
        self.root = Path(root)
        if str(self.root).startswith("/tmp"):
            raise RuntimeError(f"Formal FedBN state must not be stored under /tmp: {self.root}")
        self.expected_clients = int(expected_clients)
        self.root.mkdir(parents=True, exist_ok=True)

    def client_dir(self, client_id: int) -> Path:
        return self.root / f"client_{int(client_id)}"

    def latest_path(self, client_id: int) -> Path:
        return self.client_dir(client_id) / "latest.pt"

    def save(self, client_id: int, round_num: int, state: Mapping[str, torch.Tensor]) -> Path:
        path = self.latest_path(client_id)
        payload = {
            "client_id": int(client_id),
            "round": int(round_num),
            "state": dict(state),
        }
        _atomic_torch_save(payload, path)
        _atomic_json_save(
            {
                "client_id": int(client_id),
                "round": int(round_num),
                "num_tensors": len(state),
                "sha256": sha256_file(path),
                "path": str(path),
            },
            self.client_dir(client_id) / "latest.json",
        )
        return path

    def load(self, client_id: int) -> tuple[dict[str, torch.Tensor] | None, int]:
        path = self.latest_path(client_id)
        if not path.is_file():
            return None, 0
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["client_id"]) != int(client_id):
            raise RuntimeError(f"FedBN client-id mismatch in {path}")
        return payload["state"], int(payload["round"])

    def collect(self, required_round: int | None = None) -> dict[int, dict[str, torch.Tensor]]:
        result: dict[int, dict[str, torch.Tensor]] = {}
        for client_id in range(self.expected_clients):
            state, state_round = self.load(client_id)
            if state is None:
                raise RuntimeError(f"Missing FedBN state for client {client_id} under {self.root}")
            if required_round is not None and state_round != int(required_round):
                raise RuntimeError(
                    f"Stale FedBN state for client {client_id}: round={state_round}, "
                    f"required={required_round}"
                )
            result[client_id] = state
        return result


def save_initial_local_state(model: torch.nn.Module, path: str | Path) -> Path:
    path = Path(path)
    _atomic_torch_save(
        {"state": capture_local_state(model), "layout": layout(model)},
        path,
    )
    return path


def load_initial_local_state(model: torch.nn.Module, path: str | Path) -> None:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    load_local_state(model, payload["state"])


def arrays_to_torch(arrays: ArrayRecord) -> dict[str, torch.Tensor]:
    return arrays.to_torch_state_dict()


def strategy_state(strategy: Any) -> dict[str, Any]:
    return {
        "eta": float(strategy.eta),
        "eta_l": float(strategy.eta_l),
        "beta_1": float(strategy.beta_1),
        "beta_2": float(strategy.beta_2),
        "tau": float(strategy.tau),
        "current_arrays": strategy.current_arrays,
        "m_t": strategy.m_t,
        "v_t": strategy.v_t,
    }


def restore_strategy_state(strategy: Any, state: Mapping[str, Any]) -> None:
    strategy.current_arrays = state.get("current_arrays")
    strategy.m_t = state.get("m_t")
    strategy.v_t = state.get("v_t")


def save_fedbn_bundle(
    path: str | Path,
    *,
    shared_arrays: ArrayRecord,
    store: FedBNStateStore,
    strategy: Any,
    round_num: int,
    metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Path:
    path = Path(path)
    payload = {
        "schema": "fedbn_fedyogi_bundle_v1",
        "round": int(round_num),
        "shared_state": arrays_to_torch(shared_arrays),
        "client_local_state": store.collect(required_round=int(round_num)),
        "server_optimizer_state": strategy_state(strategy),
        "metrics": dict(metrics),
        "config": dict(config),
    }
    _atomic_torch_save(payload, path)
    _atomic_json_save(
        {
            "schema": payload["schema"],
            "round": int(round_num),
            "clients": sorted(payload["client_local_state"]),
            "shared_tensors": len(payload["shared_state"]),
            "local_tensors_per_client": {
                str(k): len(v) for k, v in payload["client_local_state"].items()
            },
            "metrics": dict(metrics),
            "sha256": sha256_file(path),
            "path": str(path),
        },
        path.with_suffix(".json"),
    )
    return path
