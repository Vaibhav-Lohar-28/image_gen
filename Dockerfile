# =============================================================================
# runpod-worker-comfy — Krea 2 Identity Edit / Head Swap edition
# Customized for "Lonecat's Krea2 Identity Edit & Head Swap" workflow V6.0.2
# (lonecatsKrea2Identity_v602HeadSwapRMBG.json)
# =============================================================================

# Stage 1: Base image with ComfyUI + RunPod handler + custom nodes
# CUDA 12.8.1 — the version current ComfyUI releases run best on
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS base

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

# Install ComfyUI (v0.34.0 = native Krea 2 support + ImageStitch/Math core nodes)
# Torch is installed with cu128 wheels to match the CUDA 12.8 base image
RUN /usr/bin/yes | comfy --workspace /comfyui install --version 0.34.0 --cuda-version 12.8 --nvidia

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

RUN pip install --no-cache-dir opencv-python-headless transformers timm

# Start container
CMD ["/start.sh"]

# =============================================================================
# Stage 2: Download the models needed by the workflow
# =============================================================================
FROM base AS downloader

# --- Credentials (all optional / can stay blank) -----------------------------
# Hugging Face token: only needed for gated repos. The default mirrors below
# are public, so you can leave it blank.
ARG HUGGINGFACE_ACCESS_TOKEN=""
# Civitai API token: some Civitai downloads benefit from a token. Optional.
ARG CIVITAI_API_TOKEN="309b095068c86e2c40756380f97199c8"

# --- Direct download links (override if files move) --------------------------
# Lenovo UltraReal LoRA, Krea 2 version (civitai.com/models/1662740?modelVersionId=3075606)
ARG LENOVO_LORA_URL="https://civitai.com/api/download/models/3075606"
# Animosity Krea2 checkpoint (civitai.com/models/2596298, version "Krea2_V1.0_int8_convrot").
# Civitai does not allow hot-linking this file reliably without the exact
# version id, so paste the URL of its Download button here
# (https://civitai.com/api/download/models/<versionId>), or simply copy the
# file you already have into ./downloads/diffusion_models/Krea 2/Your models/
# before building (the downloads/ folder is merged into the image below).
ARG ANIMOSITY_URL=""

SHELL ["/bin/bash", "-c"]

WORKDIR /comfyui

RUN mkdir -p \
    "models/diffusion_models/Krea 2/Your models" \
    "models/text_encoders" \
    "models/vae" \
    "models/loras/Krea 2/Utilities" \
    "models/loras/Krea 2/Realism helpers" \
    "models/upscale_models" \
    "models/RMBG/RMBG-2.0"

# Downloads automatically use the Hugging Face token when it is provided.
RUN AUTH=""; \
    if [ -n "${HUGGINGFACE_ACCESS_TOKEN}" ]; then AUTH="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}"; fi; \
    dl() { echo "runpod-worker-comfy: downloading $1"; curl -fL --retry 3 --retry-delay 5 ${AUTH:+--header "$AUTH"} -o "$1" "$2"; }; \
    dl models/text_encoders/qwen3vl_4b_fp8_scaled.safetensors \
       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors" && \
    dl models/vae/krea2RealVae_v10.safetensors \
       "https://huggingface.co/martineux/altkreas/resolve/main/krea2RealVae_v10.safetensors" && \
    dl "models/loras/Krea 2/Utilities/krea2_identity_edit_v1_2.safetensors" \
       "https://huggingface.co/conradlocke/krea2-identity-edit/resolve/main/krea2_identity_edit_v1_2.safetensors" && \
    dl models/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth \
       "https://objectstorage.us-phoenix-1.oraclecloud.com/n/ax6ygfvpvzka/b/open-modeldb-files/o/1x-ITF-SkinDiffDetail-Lite-v1.pth" && \
    dl models/RMBG/RMBG-2.0/config.json \
       "https://huggingface.co/1038lab/RMBG-2.0/resolve/main/config.json" && \
    dl models/RMBG/RMBG-2.0/model.safetensors \
       "https://huggingface.co/1038lab/RMBG-2.0/resolve/main/model.safetensors" && \
    dl models/RMBG/RMBG-2.0/birefnet.py \
       "https://huggingface.co/1038lab/RMBG-2.0/resolve/main/birefnet.py" && \
    dl models/RMBG/RMBG-2.0/BiRefNet_config.py \
       "https://huggingface.co/1038lab/RMBG-2.0/resolve/main/BiRefNet_config.py"

# Lenovo UltraReal Krea 2 LoRA from Civitai (optional realism LoRA in the
# workflow's Power Lora Loader; shipped switched OFF, but the file must exist)
RUN CT=""; \
    if [ -n "${CIVITAI_API_TOKEN}" ]; then CT="?token=${CIVITAI_API_TOKEN}"; fi; \
    curl -fL --retry 3 --retry-delay 5 \
      -o "models/loras/Krea 2/Realism helpers/lenovo_krea2.safetensors" \
      "${LENOVO_LORA_URL}${CT}"

# Animosity Krea2 int8_convrot checkpoint from Civitai.
# Skipped (with a warning) when ANIMOSITY_URL is blank — in that case provide
# the file via the downloads/ folder (merged in stage 3) or a network volume.
RUN if [ -n "${ANIMOSITY_URL}" ]; then \
        CT=""; \
        if [ -n "${CIVITAI_API_TOKEN}" ]; then \
          case "${ANIMOSITY_URL}" in *\?*) CT="&token=${CIVITAI_API_TOKEN}";; *) CT="?token=${CIVITAI_API_TOKEN}";; esac; \
        fi; \
        curl -fL --retry 3 --retry-delay 5 \
          -o "models/diffusion_models/Krea 2/Your models/Animosity_Krea2_V1.0_int8_convrot.safetensors" \
          "${ANIMOSITY_URL}${CT}"; \
    else \
        echo ""; \
        echo "*******************************************************************"; \
        echo "WARNING: ANIMOSITY_URL is blank - the Animosity Krea2 checkpoint"; \
        echo "was NOT baked into this image. Provide it by either:"; \
        echo "  1. rebuilding with --build-arg ANIMOSITY_URL=<civitai download url>"; \
        echo "  2. placing the file at downloads/diffusion_models/Krea 2/Your models/"; \
        echo "     inside the build context before building"; \
        echo "  3. mounting a network volume that contains it"; \
        echo "*******************************************************************"; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models

# Merge any locally provided model files (e.g. large Civitai checkpoints you
# already have). Layout mirrors ComfyUI's models/ folder, e.g.:
#   downloads/diffusion_models/Krea 2/Your models/Animosity_Krea2_V1.0_int8_convrot.safetensors
COPY downloads/ /comfyui/models/

# Start container
CMD ["/start.sh"]