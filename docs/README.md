# Odin-ML Documentation

## Structure

Documentation in this repository is split by purpose:

| Path | Contains |
| :--- | :--- |
| `docs/standards/` | Enforceable engineering and documentation standards for this repository (Python/ML). |
| `docs/models/` | Model candidate roster (RRL-grounded), Classification V2 specification, model-lifecycle guidance (see `models/README.md`), and `TEAMMATE-GUIDE.md` (trusted path for new agents/humans). |
| `training/docs/` | ML design documents produced by the training pipeline (data collection, EDA, dimension discovery). |
| `training/docs/phases/` | Model development runbooks: Phase 7 evaluation, 8 selection/versioning, 9 deployment, 10 monitoring. |

## Quick start for new contributors

New contributors (human or agent) should read `docs/models/TEAMMATE-GUIDE.md` first:
it covers the decided model scope, per-family training runbooks, CPU/GPU toggles,
the metadata/eval contract, and verification commands.

## Conventions

- Standards that govern code quality live in `docs/standards/`.
- Model scope and artifacts follow the guidance in `docs/models/` and top-level `models/README.md`.
- Generated ML design reports (EDA, dimension-threshold candidates, data collection) live under `training/docs/` alongside the artifacts they describe.
- Follow the shared formatting rules in `docs/standards/documentation-format.md`.
