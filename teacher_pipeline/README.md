# AeroJAX frozen teacher dataset

Production numerical code and GUI are unchanged. Entry point:
`BaselineSolver._step_jit -> _step_collocated -> rk_step_unified -> poisson_multigrid`.
Collocated/RK3/legacy 5 V-cycles, zero initial pressure guess, legacy coarse spacing.
Offline export uses fixed dt, with exactly the production numerical step; it does not run viewer,
adaptive-dt controller or scalar transport. AUTO-DT in the GUI is unchanged.

## Run

From `~/AeroJAX` after installing this folder as `teacher_pipeline`:

```bash
python -m pip install --no-deps --target /tmp/aerojax-dataset-deps -r teacher_pipeline/requirements-dataset.txt
PYTHONPATH=/tmp/aerojax-dataset-deps XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ~/aerojax_env_clean/bin/python teacher_pipeline/teacher_dataset.py generate \
  --config teacher_pipeline/pilot.json --output /path/to/teacher-pilot
```

Use the same command to resume. `--frame-budget 5 --limit 1` is a bounded interruption test.
`inspect --output /path/to/teacher-pilot` rebuilds QC and training-only normalization.
GPU is required; CPU fallback is an error. Multiple independent output roots can run independent
campaigns; a filesystem lock prevents concurrent writers to one campaign. Do not train while writing.

## Schema v1

One HDF5 file per sequence, under `train/`, `validation/`, or `test/`:

* `fields`: float32 `[T,3,nx,ny]`, channels u/v/p, chunk `[1,3,nx,ny]`, LZF + shuffle + Fletcher32.
* `mask`, `sdf`: float32 `[nx,ny]`, stored once; 1=fluid, SDF positive in fluid.
* `x`, `y`: actual teacher coordinates, stored separately from operator dx/dy.
* `time`, `dt`: float64 `[T]`; dt is the CFD step, **not** the interval between stored frames.
* `iteration`, `frame_index`: int64 `[T]`.
* Attributes `case`, `resolved`, `source_sha256`, `case_hash`, `committed`, `status`, timings.
* `case` contains seed, sequence_id, geometry.id (=geometry_id), family, split, geometry parameters,
  requested U/Re/nu, AoA, grid, warmup, stride and frames.
* `resolved` contains actual U/nu/Re/L, pressure gauge, inlet direction, resolved solver controls,
  geometry hash and whether SDF is native or derived from the raster.
* Derived raster SDF is metadata only; it is never fed back into teacher penalization.
* Vorticity is optional/derivable using the teacher operators; no scalar field in v1 pilot.

Writer appends payload, flushes, then updates the committed-frame marker. Restart trims uncommitted
tails and reconstructs u/v/p, iteration and physical time from the last committed frame.
SIGINT/SIGTERM close files cleanly. Sudden power loss/SIGKILL while HDF5 metadata is being written
is not a transactional storage guarantee; quarantine a corrupt file and regenerate that sequence.
Nonfinite frames are rejected before commit; failed cases are logged and do not stop later cases.
Changed case/source cannot resume. Asset bytes are hashed when external PNG/mask files are used.

## Geometry and controls

Native NACA and cylinder paths are used. Procedural ellipse/rectangle/solid_wall and imported
PNG/mask/Tesla use the existing production custom-mask path (no new physics).
PNG: white fluid, black solid, threshold 128, nearest resize and transpose to `[nx,ny]`.
NumPy mask: finite float `[nx,ny]` in [0,1], no pickle. Tesla defaults to the existing repository asset.
New families can supply a PNG or raster mask and a leakage_group.
`solid_wall` denotes an internal wall obstacle, not a new external BC type.
Direction is currently horizontal. Nonzero inlet angles are rejected explicitly; NACA/procedural
AoA rotates geometry, not the boundary velocity. Native NACA chord is resolved by the teacher to 15% lx.
Custom-shape Reynolds length is the configured `2*radius` convention (default 1); always use resolved metadata.

## Splits and loading

Pilot: NACA/cylinder train, ellipses validation, rectangles test. All parameter changes of a geometry
remain together. Entire families are grouped to prevent near-duplicate leakage. User-defined leakage_group
can group further related assets. QC rejects identical mask hashes across splits; similarity inspection
remains required for new imported assets from unrelated source labels.

`loader.TeacherPairs` is a map-style NumPy reader compatible with a future PyTorch DataLoader.
It emits state/target/mask/SDF/parameters/IDs and never crosses sequence boundaries. HDF5 is opened
per read, so handles are not inherited by workers. Normalization comes from training only. PyTorch
and PhysicsNeMo are not installed or required to generate/read NumPy samples.

## Format decision

HDF5 supports compression, resizing and partial reads in a bounded number of local files, suitable
for WSL ext4 and single-writer generation. Zarr v3 is preferable for object storage/distributed writing;
sharding avoids many small chunk objects, but adds a chunk/shard tuning decision. Revisit for distributed
training; the logical schema is independent of storage. This is a suitability decision, not a measured
HDF5-vs-Zarr speed claim. Do not store large datasets on Windows-mounted NTFS for production throughput.

## Teacher limitations

This dataset learns this numerical teacher, not certified physical ground truth. Retain the existing
reports on finite pressure residual, projection/operator mismatch, grid-coordinate spacing, temporal
and spatial error and transient cavity tests. Five cycles is the compatibility baseline, not a guarantee
of converged high-accuracy Poisson. Pilot low resolution and short physical duration cannot certify
shedding statistics, steady flow or unseen arbitrary geometries. No solver audit is reopened here.

## Future GUI contract

Keep fields u/v/p/mask/SDF in the same orientation and physical units. A future backend interface exposes
reset(geometry, resolved_params, seed), advance_to(time), snapshot(), status(), close(). CFD and Neural
backends implement it; worker owns the backend and viewer consumes immutable snapshots. Switching backend
resets rollout/history or explicitly initializes from a CFD snapshot. Display backend/model version,
training domain and physical time. Unsupported BC, geometry or parameter range requires CFD selection.
Neural wall latency and neural RTF must be measured separately. No GUI code is changed in this delivery.

## Training plan

First model: geometry-conditioned autoregressive FNO, four layers, width 32, modes [16,12], explicit
nonperiodic padding and BC channels. Condition on interval between frames. Start one-step, then 2/4/8-step
rollouts; evaluate 100/1000-step free rollout on held-out geometry, field/energy/force/frequency diagnostics.
Compare an inexpensive U-Net baseline. AFNO is a later patch-token benchmark if grids become much larger;
latent rollout adds reconstruction drift and is premature. Geometry+parameters+time alone cannot represent
different initial states/interactive changes and is not the first formulation. PhysicsNeMo GINO/DoMINO-type
geometry operators become relevant for point clouds/unstructured or larger geometry campaigns, rather than
adding that complexity to today's fixed Cartesian fields. No training or inference speed is claimed yet.

Sources checked 2026-09-09:
https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api_models.html
https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/_modules/physicsnemo/models/fno/fno.html
https://github.com/NVIDIA/physicsnemo/blob/main/examples/README.md
https://arxiv.org/abs/2111.13587
https://arxiv.org/abs/2309.00583
https://docs.h5py.org/en/stable/
https://zarr.readthedocs.io/en/v3.0.10/user-guide/performance.html
