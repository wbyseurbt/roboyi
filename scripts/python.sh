#!/usr/bin/env bash
# Run the project venv with Isaac Sim's library paths and Python 3.11 runtime.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_dir="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
if [[ ! -x "$project_dir/.venv/bin/python" ]]; then
    echo "Project environment missing. Run ./scripts/setup_env.sh first." >&2
    exit 1
fi
export PYTHONEXE="$project_dir/.venv/bin/python"
export PYTHONUNBUFFERED=1
# Isaac Sim otherwise puts its preinstalled RSL-RL ahead of editable packages.
export PYTHONPATH="$project_dir/robolab:$project_dir/rsl_rl:$project_dir/.venv/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "$sim_dir/python.sh" "$@"
