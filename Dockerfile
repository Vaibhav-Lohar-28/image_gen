# Stage 1: Base image with ComfyUI + RunPod handler
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    libgl1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install comfy-cli
RUN pip install comfy-cli

# Install ComfyUI (v0.34.0 = native Krea 2 support, as required by the
# "Lonecat's Krea2 Identity Edit & Head Swap" workflow)
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.34.0 --cuda-version 12.6 --nvidia

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install the RunPod handler dependencies
RUN pip install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add scripts
ADD src/start.sh src/restore_snapshot.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh /restore_snapshot.sh

# Add the snapshot file (custom nodes needed by the workflow)
ADD *snapshot*.json /

# Restore the snapshot to install custom nodes
RUN /restore_snapshot.sh

# Start container
CMD ["/start.sh"]

# Stage 2: Download the models needed by the Krea 2 Identity Edit workflow
FROM base AS downloader

# Your Hugging Face token.
# Leave it blank when downloading from the public mirrors used below (Comfy-Org/Krea-2).
# Fill it in (docker build --build-arg HUGGINGFACE_ACCESS_TOKEN=hf_xxx) when you
# download from gated repositories such as https://huggingface.co/krea/Krea-2-Turbo.
ARG HUGGINGFACE_ACCESS_TOKEN=""

SHELL ["/bin/bash", "-c"]

WORKDIR /comfyui

RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras/Krea2

# Downloads use the Hugging Face token automatically when it is provided.
RUN AUTH=""; \
    if [ -n "${HUGGINGFACE_ACCESS_TOKEN}" ]; then AUTH="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}"; fi; \
    dl() { echo "runpod-worker-comfy: downloading $1"; curl -fL --retry 3 --retry-delay 5 ${AUTH:+--header "$AUTH"} -o "$1" "$2"; }; \
    dl models/diffusion_models/krea2_turbo_fp8_scaled.safetensors \
       https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors && \
    dl models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
       https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors && \
    dl models/vae/qwen_image_vae.safetensors \
       https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors && \
    dl models/loras/Krea2/krea2_identity_edit_v1_2.safetensors \
       https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors

# Optional: Krea 2 Raw (fp8) — used for removal/deletion edits at CFG ~3.
# Uncomment to bake it into the image (adds ~13 GB):
# RUN AUTH=""; \
#     if [ -n "${HUGGINGFACE_ACCESS_TOKEN}" ]; then AUTH="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}"; fi; \
#     curl -fL --retry 3 --retry-delay 5 ${AUTH:+--header "$AUTH"} \
#       -o models/diffusion_models/krea2_raw_fp8_scaled.safetensors \
#       https://huggingface.co/Comfy-Org/Krea-2/resolve/main/diffusion_models/krea2_raw_fp8_scaled.safetensors

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Start container
CMD ["/start.sh"]
