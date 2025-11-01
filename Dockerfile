# 1) Pick the CUDA/cuDNN that matches your working setup
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# 2) Basics
SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bzip2 git tini \
 && rm -rf /var/lib/apt/lists/*

# 3) Micromamba (fast/lean Conda)
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV MAMBA_DOCKERFILE_ACTIVATE=1
RUN curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ bin/micromamba --strip-components=1

# 4) Create env from your environment.yml (pin exact versions there)
WORKDIR /tmp
COPY environment.yml .
RUN micromamba create -y -n ml -f environment.yml && \
    micromamba clean --all --yes

# 5) App code
WORKDIR /app
COPY . /app

# 6) Default to running inside the env; pass your CLI args after `docker run ...`
ENTRYPOINT ["/usr/bin/tini", "--", "micromamba", "run", "-n", "ml"]
CMD ["bash"]
