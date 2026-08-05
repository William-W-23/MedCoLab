"""Active MedCoLab task configuration and task-level public API.

Change ``ACTIVE_TASK`` to ``"classification"`` to expose the classification
task through this module. Task-specific implementations remain separate so
that detection and classification code from the two audited servers cannot
silently overwrite one another.
"""

ACTIVE_TASK = "detection"

if ACTIVE_TASK == "detection":
    from fl.detection_task import *  # noqa: F401,F403
elif ACTIVE_TASK == "classification":
    from fl.classification_task import *  # noqa: F401,F403
else:
    raise RuntimeError(f"Unsupported ACTIVE_TASK={ACTIVE_TASK!r}")
