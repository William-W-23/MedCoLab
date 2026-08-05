#!/usr/bin/env python3
"""Validate and summarize one completed independent SSL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "ssl_global_backbone.pt",
    "ssl_full_moco_last.pt",
    "ssl_best_by_loss_backbone.pt",
    "ssl_best_by_loss_full.pt",
    "ssl_best_by_loss_meta.json",
    "ssl_metrics_history.json",
    "ssl_meta.json",
    "server_result.json",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (args.output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing SSL artifacts: {missing}")

    history = json.loads((args.output / "ssl_metrics_history.json").read_text())
    valid = [row for row in history if row.get("ssl_loss") is not None]
    if not valid:
        raise RuntimeError("No finite SSL loss history")
    best = min(valid, key=lambda row: float(row["ssl_loss"]))
    final = valid[-1]
    meta = json.loads((args.output / "ssl_meta.json").read_text())
    summary = {
        "source_client": meta.get("config", {}).get("source_client"),
        "master_seed": meta.get("config", {}).get("master_seed"),
        "manifest_sha256": meta.get("config", {}).get("data_manifest_sha256"),
        "rounds": len(valid),
        "best_round": int(best["round"]),
        "best_ssl_loss": float(best["ssl_loss"]),
        "final_round": int(final["round"]),
        "final_ssl_loss": float(final["ssl_loss"]),
    }
    (args.output / "ssl_summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "ssl_summary_metrics.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in valid]
    losses = [float(row["ssl_loss"]) for row in valid]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(rounds, losses, color="#1769aa", linewidth=1.8)
    axis.scatter([best["round"]], [best["ssl_loss"]], color="#c62828", s=36, zorder=3)
    axis.set_xlabel("Round")
    axis.set_ylabel("SSL loss")
    axis.set_title(f"MoCo SSL convergence - {summary['source_client']}")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output / "ssl_convergence_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
