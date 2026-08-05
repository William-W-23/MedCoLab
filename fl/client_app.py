"""Flower client entrypoint for the active MedCoLab task."""

from fl.task import ACTIVE_TASK

if ACTIVE_TASK == "detection":
    from fl.detection_client_app import app
elif ACTIVE_TASK == "classification":
    from fl.classification_client_app import app
else:
    raise RuntimeError(f"Unsupported ACTIVE_TASK={ACTIVE_TASK!r}")

__all__ = ["app"]
