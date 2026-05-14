# chip-check

## Setup
```bash
git clone https://github.com/Dev077/chip-check
cd chip-check
git submodule update --init --recursive
```

## AutoDMP Docker Setup

AutoDMP is not a Python package, so it cannot be installed with `uv add`. It is a separate C++/CUDA project that ships its own Dockerfile in the upstream repository.

Run the helper script from this repository to clone and build the AutoDMP image:

```bash
bash scripts/setup_autodmp_docker.sh
```

If you prefer to do it manually, use:

```bash
git clone https://github.com/NVlabs/AutoDMP.git external/AutoDMP
cd external/AutoDMP
docker build --no-cache --tag autodmp:latest .
```

After the image is built, you can start the container with:

```bash
bash scripts/run_autodmp_docker.sh
```

Set `AUTO_DMP_USE_GPU=1` if your Docker setup exposes NVIDIA GPUs and you want the container launched with `--gpus all`.

Then run the container with this repository mounted so the AutoDMP scripts can access the MacroPlacement data and testcases:

```bash
docker run --rm -it --gpus all \
	-v /Users/devchaudhari/Documents/GitHub/chip-check:/workdir/chip-check \
	-w /workdir/chip-check \
	autodmp:latest bash
```

Inside the container, use the paths under `external/MacroPlacement` as the MacroPlacement source tree.
