from pathlib import Path
import ast
import re
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_PARTS = {
    ".git", "__pycache__", ".venv", "venv",
    "aerojax_env", "aerojax_env_clean",
}

def valid_file(p):
    if p.suffix != ".py":
        return False
    if any(x in EXCLUDE_PARTS for x in p.parts):
        return False
    if ".bak" in p.name:
        return False
    return True

files = [p for p in ROOT.rglob("*.py") if valid_file(p)]

print("=" * 78)
print("AEROJAX AUDIT")
print("=" * 78)
print(f"Python files: {len(files)}")
print()

# ----------------------------------------------------------------------
# 1. Syntax / AST
# ----------------------------------------------------------------------
print("[1] SYNTAX / DUPLICATE DEFINITIONS")
print("-" * 78)

syntax_errors = []
duplicate_defs = []

for p in files:
    try:
        text = p.read_text(errors="replace")
        tree = ast.parse(text, filename=str(p))
    except Exception as e:
        syntax_errors.append((p, e))
        continue

    scopes = defaultdict(list)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scopes[node.name].append(node.lineno)

    for name, lines in scopes.items():
        if len(lines) > 1:
            # Some same-name methods in different classes are legitimate.
            # Report them as candidates rather than errors.
            duplicate_defs.append((p, name, lines))

if syntax_errors:
    for p, e in syntax_errors:
        print(f"ERROR  {p.relative_to(ROOT)}: {e}")
else:
    print("OK     No syntax errors detected")

print()
print("Duplicate-name candidates:")
for p, name, lines in duplicate_defs[:80]:
    print(f"  {p.relative_to(ROOT)} :: {name} -> {lines}")

# ----------------------------------------------------------------------
# 2. Runtime/thread hotspots
# ----------------------------------------------------------------------
print()
print("[2] THREAD / WORKER HOTSPOTS")
print("-" * 78)

patterns = {
    "thread_start": r"\.start\(\)",
    "metrics_worker": r"MetricsWorker|metrics_worker",
    "data_ready_emit": r"data_ready\.emit",
    "signal_emit": r"\.emit\(",
    "while_running": r"while\s+(?:self\.)?running",
    "sleep": r"time\.sleep",
}

for label, pat in patterns.items():
    hits = []
    rx = re.compile(pat)

    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append((p, n, line.strip()))

    print(f"\n{label}: {len(hits)}")
    for p, n, line in hits[:50]:
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 3. Shared memory / cleanup
# ----------------------------------------------------------------------
print()
print("[3] SHARED MEMORY / CLEANUP")
print("-" * 78)

patterns = [
    r"SharedMemory",
    r"shared_memory",
    r"SharedData",
    r"\.cleanup\(",
    r"\.unlink\(",
    r"\.close\(",
]

for pat in patterns:
    rx = re.compile(pat)
    hits = []

    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append((p, n, line.strip()))

    print(f"\n{pat}: {len(hits)}")
    for p, n, line in hits[:60]:
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 4. Qt layout problems
# ----------------------------------------------------------------------
print()
print("[4] QT LAYOUT")
print("-" * 78)

layout_hits = []

for p in files:
    lines = p.read_text(errors="replace").splitlines()

    for n, line in enumerate(lines, 1):
        if "addWidget(" in line or "addLayout(" in line:
            layout_hits.append((p, n, line.strip()))

for p, n, line in layout_hits:
    if any(x in str(p) for x in ("viewer", "ui_components", "main.py")):
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 5. JAX performance hazards
# ----------------------------------------------------------------------
print()
print("[5] JAX / GPU PERFORMANCE HOTSPOTS")
print("-" * 78)

jax_patterns = {
    "jax.clear_caches": r"jax\.clear_caches",
    "np.array(JAX)": r"np\.array\(",
    "np.asarray": r"np\.asarray\(",
    "device_get": r"device_get",
    "block_until_ready": r"block_until_ready",
    "jit_compile": r"get_step_jit|jax\.jit",
}

for label, pat in jax_patterns.items():
    rx = re.compile(pat)
    hits = []

    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append((p, n, line.strip()))

    print(f"\n{label}: {len(hits)}")
    for p, n, line in hits[:50]:
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 6. Timestep / Reynolds logic
# ----------------------------------------------------------------------
print()
print("[6] TIMESTEP / REYNOLDS")
print("-" * 78)

patterns = {
    "dt assignments": r"(?:self\.solver\.dt|self\.dt)\s*=",
    "dt_min": r"dt_min",
    "adaptive_dt": r"adaptive_dt",
    "CFL": r"cfl|CFL",
    "Reynolds": r"Reynolds|reynolds|new_Re|lock_Re",
}

for label, pat in patterns.items():
    rx = re.compile(pat)
    hits = []

    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append((p, n, line.strip()))

    print(f"\n{label}: {len(hits)}")
    for p, n, line in hits[:80]:
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 7. Store / duplicate-state architecture
# ----------------------------------------------------------------------
print()
print("[7] STATE / STORE ARCHITECTURE")
print("-" * 78)

patterns = {
    "dispatch": r"\.dispatch\(",
    "SET_ actions": r"SET_[A-Z_]+",
    "latest_data": r"latest_data",
    "latest_metrics": r"latest_metrics",
    "solver state writes": r"self\.solver\.[A-Za-z_]+\s*=",
}

for label, pat in patterns.items():
    rx = re.compile(pat)
    hits = []

    for p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if rx.search(line):
                hits.append((p, n, line.strip()))

    print(f"\n{label}: {len(hits)}")
    for p, n, line in hits[:70]:
        print(f"  {p.relative_to(ROOT)}:{n}: {line}")

# ----------------------------------------------------------------------
# 8. Backup pollution
# ----------------------------------------------------------------------
print()
print("[8] BACKUP / DEAD CODE")
print("-" * 78)

bak_files = list(ROOT.rglob("*.bak*"))

for p in bak_files[:100]:
    try:
        print(" ", p.relative_to(ROOT))
    except Exception:
        print(" ", p)

print()
print("=" * 78)
print("AUDIT COMPLETE")
print("=" * 78)
