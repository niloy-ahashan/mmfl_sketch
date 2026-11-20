#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FedMobile/torch_port/run_ptbxl_p44.sh" "$@"
