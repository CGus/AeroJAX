# Repository cleanup inventory — 2026-09-13

This inventory records the conservative cleanup performed after the CFD audit and PhysicsNeMo FNO integration. No production change was discarded.

## Production code retained

- `main.py`, `solver/`, `pressure_solvers/`, `viewer/`, `advection_schemes/`, `obstacles/`, `lbm/`, `timestepping/` and the existing auxiliary application packages.
- All 25 previously modified tracked files. Their changes include CFD stability/audit work, GUI behavior and the FNO integration points; none were reverted.
- `assets/tesla_valve_mask.png`, which is referenced by production mask generation and the teacher pipeline.

## PhysicsNeMo integration retained

- `viewer/physicsnemo_backend.py`.
- `physicsnemo_fno/inference_server.py`.
- FNO training/evaluation scripts, resolved configurations, environment manifests and technical reports.
- Local `.venv-physicsnemo/`, ignored by Git because it is reproducible from the lock/freeze files.
- Local model checkpoints under `physicsnemo_fno/runs/`, ignored by Git. The GUI-required surrogate v1 remains at `physicsnemo_fno/runs/residual_fno/best.pt`.

## Validation tools and historical material retained

- `tools/`: CFD audit, validation, profiling and pressure-solver benchmarks.
- `teacher_pipeline/`: frozen teacher dataset generation, loading and integrity tests.
- `physicsnemo_fno/*.md`, including pilot, residual and GUI integration reports.
- JSON metrics and generated plots under `physicsnemo_fno/runs/`, retained locally but ignored as generated run output.
- Long-evaluation HDF5 files under `physicsnemo_fno/datasets/`, retained locally but ignored as dataset artifacts.

## Recoverable archive outside the repository

The following categories were moved to `/home/gus/AeroJAX-local-archive/2026-09-13-repository-cleanup/`:

- all `*.bak*` source snapshots;
- one-off patch scripts `fix_interface.py` and `fix_sidebar_width.py`;
- empty accidental shell-redirection files: `PY`, `else:`, `from`, `if`, `new`, `old`;
- runtime CSV and audit text dumps;
- unreferenced generated Tesla-valve and virtual-wind-tunnel preview PNGs.

Nothing in this archive was deleted. Python bytecode caches remain in place but are ignored and reproducible.

## Classification summary

| Category | Repository treatment |
|---|---|
| Production code | retained, intended for source control |
| PhysicsNeMo bridge and scripts | retained, intended for source control |
| Test/validation tools | retained, intended for source control |
| Technical reports/configuration | retained, intended for source control |
| HDF5 datasets | retained locally, ignored |
| `.pt/.pth/.ckpt` checkpoints | retained locally, ignored |
| Run metrics and plots | retained locally under ignored run directories |
| Virtual environment and caches | retained locally or reproducible, ignored |
| Backups/logs/temporary output | moved to recoverable external archive |

The recommended commit split is: (1) CFD audit and stability fixes plus validation tools, (2) PhysicsNeMo surrogate sources and GUI integration, and (3) `.gitignore`, this inventory and repository hygiene. Because the integration edits overlap several already-modified GUI files, commits should be staged by path and, for overlapping files, by reviewed hunks.
