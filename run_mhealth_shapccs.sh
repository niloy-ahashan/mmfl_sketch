#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/ShapCCS/torch_port/run_mhealth.sh" "$@"
