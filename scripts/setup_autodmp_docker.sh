#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
autodmp_dir="$repo_root/external/AutoDMP"
image_name="autodmp:latest"

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

if [[ ! -d "$autodmp_dir/.git" ]]; then
  mkdir -p "$repo_root/external"
  git clone https://github.com/NVlabs/AutoDMP.git "$autodmp_dir"
fi

if ! "$docker_bin" info >/dev/null 2>&1; then
  cat <<'EOF'
Docker is installed, but the Docker daemon is not running or is not reachable.

Start Docker Desktop for macOS, or start your Docker daemon/engine, then rerun this script.
EOF
  exit 1
fi

# Use the pre-patched Dockerfile from the repo that uses HTTPS instead of SSH
dockerfile_path="$repo_root/Dockerfile.autodmp"

"$docker_bin" build --no-cache -f "$dockerfile_path" --tag "$image_name" "$repo_root"

if [[ $? -ne 0 ]]; then
  echo "Docker build failed. Check the logs above for details."
  exit 1
fi

cat <<EOF

AutoDMP image is ready.

To start an interactive container with this repo mounted:

"$docker_bin" run --rm -it --gpus all \
  -v "$repo_root":/workdir/chip-check \
  -w /workdir/chip-check \
  "$image_name" bash
EOF