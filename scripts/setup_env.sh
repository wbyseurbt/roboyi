#!/usr/bin/env bash
# Use the existing Isaac Sim / Isaac Lab installation, as in README_CN.md.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sim_dir="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
cd "$project_dir"
# The machine's proxy truncates large wheels; use the configured PyPI mirror directly.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}pypi.tuna.tsinghua.edu.cn"
git submodule update --init --recursive
if [[ ! -x .venv/bin/python ]]; then
    "$sim_dir/python.sh" -m venv --system-site-packages .venv
fi
./scripts/python.sh -m pip install --use-pep517 --timeout 120 \
    -c constraints-isaacsim51.txt -e ./robolab -e ./rsl_rl
