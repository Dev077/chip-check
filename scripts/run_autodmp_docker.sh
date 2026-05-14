#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

docker_bin="${DOCKER_BIN:-}"
if [[ -z "$docker_bin" ]]; then
  if command -v docker >/dev/null 2>&1; then
    docker_bin="$(command -v docker)"
  elif [[ -x /Applications/Docker.app/Contents/Resources/bin/docker ]]; then
    docker_bin="/Applications/Docker.app/Contents/Resources/bin/docker"
  elif [[ -x /usr/local/bin/docker ]]; then
    docker_bin="/usr/local/bin/docker"
  elif [[ -x /opt/homebrew/bin/docker ]]; then
    docker_bin="/opt/homebrew/bin/docker"
  else
    cat <<'EOF'
Docker was not found.

Install Docker Desktop for macOS or make sure the Docker CLI is on your PATH,
then rerun this script.
EOF
    exit 1
  fi
fi

if ! "$docker_bin" image inspect autodmp:latest >/dev/null 2>&1; then
  cat <<'EOF'
The autodmp:latest image is not present locally.

Run scripts/setup_autodmp_docker.sh first to build it.
EOF
  exit 1
fi

gpu_args=()
if [[ "${AUTO_DMP_USE_GPU:-0}" == "1" ]]; then
  gpu_args=(--gpus all)
fi

"$docker_bin" run --rm -it "${gpu_args[@]}" \
  -v "$repo_root":/workdir/chip-check \
  -w /workdir/chip-check \
  autodmp:latest bash