#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/MFedMC/ptbxl/run_ptbxl_mm.sh" "$@"
