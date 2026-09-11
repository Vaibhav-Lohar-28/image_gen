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


# Stage 2: Download the models needed by the workflow
# ======================================================================
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
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BodySegment.py: No module named 'onnxruntime'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Florence2.py: No module named 'matplotlib'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_ImageMaskTools.py: No module named 'cv2'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_RMBG.py: No module named 'cv2'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM2Segment.py: No module named 'hydra'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM3Segment.py: No module named 'cv2'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Warning: diffusers/transformers not available. SDMatte functionality will be limited.\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Segment.py: No module named 'segment_anything'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SegmentV2.py: No module named 'segment_anything'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_YoloV8.py: No module named 'cv2'\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [34m[ComfyUI-RMBG][0m v[93m3.1.0[0m | [93m9 nodes[0m [92mLoaded[0m\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [vsLinx] ComfyUI-Impact-Pack not found - skipping the '(Impact-Pack) Interactive Detailer' node. All other vsLinx nodes are unaffected.\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] loading from /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] LUTs → /comfyui/models/luts: 18 new file(s) copied, 0 existing skipped (no overwrite)\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] + aspect_ratio: ['AspectRatioSimplifier', 'LCAspectRatioPipeOut', 'LCAspectRatioPipe']\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] + slider: ['LCSlider']\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] + anima_regional_canvas: ['AnimaRegionalCanvasInline']\n
2026-09-11 05:34:12.418 | info | 1tftbfouxopgsy | [LC123] + krea2_regional_canvas: ['Krea2RegionalCanvasInline']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + dynamic_overlay: ['LCDynamicOverlay']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_any_switch: ['LCAnySwitch']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_index_switch: ['LCIndexSwitch']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_custom_combo: ['LCCustomCombo', 'LCCustomComboPanel']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] registered LCComboSelector\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_combo: ['LCComboSelector']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_invert_boolean: ['LCInvertBoolean']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_boolean: ['LCBoolean']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_boolean_switch: ['LCBooleanSwitch', 'LCBooleanFlip', 'LCBooleanValue']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_builder: ['LCSubject', 'LCSubjectArray', 'LCCamera', 'LCLighting', 'LCStyleSelector', 'LCSceneBuilder', 'LCColorPalette', 'LCPromptAssembler']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_join_strings: ['LCJoinStrings']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_show_text: ['LCShowText']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_widget_to_string: ['LCWidgetToString']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_text_replace: ['LCTextReplace']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_text_remove: ['LCTextRemove']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_compare: ['LCIntCompare', 'LCFloatCompare']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_seed_jump: ['LCSeedJump']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_node_snapshot: ['LCNodeSnapshot']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_notify: ['LCNotify']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_civitai_strip: ['LCCivitaiStrip']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_save_text: ['LC123SaveText']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_batch_image: ['LCBatchImage']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_batch_image_comparer: ['LCBatchImageComparer']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_image_split: ['LCImageSplit']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_last_image_holder: ['LCLastImageHolder']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_sampler_configure: ['LCSamplerConfigure', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigurePipe', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_pipe_io: ['LCPipeOut', 'LCPipeEdit', 'LCDetailPipeOut']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_minimax_h3_pipe: ['LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_get_image: ['LCGetImage']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_dimension_resize: ['LCDimensionResize']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_image_mask_resize: ['LCImageMaskResize']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_image_crop: ['LCImageCrop']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_image_grid: ['LCImageGrid']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_image_tools: ['LCImageAdjust', 'LCAutoWhiteBalance', 'LCClarity', 'LCLensFX', 'LCLiftGammaGain', 'LCImageRGB', 'LCFilmGrain', 'LCVibrance', 'LCVignette', 'LCBloom', 'LCImageDenoise', 'LCColorMatch', 'LCToneMatch', 'LCFilmStockBW', 'LCFilmStockColor', 'LCLensProfile', 'LCChromaticAberration', 'LCImageDesaturate']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_skin_beauty: ['LCSkinBeauty']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_skin_upscale: ['LCSkinUpscale']\n
2026-09-11 05:34:12.419 | info | 1tftbfouxopgsy | [LC123] + lc_phone_look: ['LCPhoneLook']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_apply_lut: ['LCApplyLUT']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_text_overlay: ['LCTextOverlay']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_watermark: ['LCWatermark']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_to_conditioning: ['LCPromptToConditioning', 'LCPromptToConditioningZero']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_split_sigma_scheduler: ['LCSplitSigmaScheduler']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_basic_scheduler: ['LCBasicScheduler']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_split_sigmas_advanced: ['LCSplitSigmasAdvanced']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_box: ['LCPositive', 'LCNegative']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_vram_cache_clear: ['LCVRAMCacheClear']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_stop: ['LCStop']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_advanced_folder: ['LCAdvancedFolder']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_easy_folder: ['LCEasyFolder']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_save_image: ['LCSaveImage', 'LCSaveImageMetadata']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_any_empty: ['LCAnyEmptyBool', 'LCAnyEmptyInt', 'LCAnyEmptyFloat']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_int_split: ['LCIntSplit']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_change_step_count: ['LCSigmaResample', 'LCChangeStepCount']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] + lc_sigma_curve: ['LCSigmaCurve']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | USDU batch patches applied successfully.\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m USDU batch patches applied successfully.\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [LC123] total 105 nodes: ['AnimaRegionalCanvasInline', 'AspectRatioSimplifier', 'Krea2RegionalCanvasInline', 'LC123SaveText', 'LCAdvancedFolder', 'LCAnyEmptyBool', 'LCAnyEmptyFloat', 'LCAnyEmptyInt', 'LCAnySwitch', 'LCApplyLUT', 'LCAspectRatioPipe', 'LCAspectRatioPipeOut', 'LCAutoWhiteBalance', 'LCBasicScheduler', 'LCBatchImage', 'LCBatchImageComparer', 'LCBloom', 'LCBoolean', 'LCBooleanFlip', 'LCBooleanSwitch', 'LCBooleanValue', 'LCCamera', 'LCChangeStepCount', 'LCChromaticAberration', 'LCCivitaiStrip', 'LCClarity', 'LCColorMatch', 'LCColorPalette', 'LCComboSelector', 'LCCustomCombo', 'LCCustomComboPanel', 'LCDenoise', 'LCDetailPipeOut', 'LCDimensionResize', 'LCDynamicOverlay', 'LCEasyFolder', 'LCFilmGrain', 'LCFilmStockBW', 'LCFilmStockColor', 'LCFloatCompare', 'LCGetImage', 'LCImageAdjust', 'LCImageCrop', 'LCImageDenoise', 'LCImageDesaturate', 'LCImageGrid', 'LCImageMaskResize', 'LCImageRGB', 'LCImageSplit', 'LCIndexSwitch', 'LCIntCompare', 'LCIntSplit', 'LCInvertBoolean', 'LCJoinStrings', 'LCLastImageHolder', 'LCLensFX', 'LCLensProfile', 'LCLiftGammaGain', 'LCLighting', 'LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut', 'LCNegative', 'LCNodeSnapshot', 'LCNotify', 'LCPhoneLook', 'LCPipeEdit', 'LCPipeOut', 'LCPositive', 'LCPromptAssembler', 'LCPromptToConditioning', 'LCPromptToConditioningZero', 'LCReferenceLatent', 'LCRelight', 'LCSamplerConfigure', 'LCSamplerConfigurePipe', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut', 'LCSaveImage', 'LCSaveImageMetadata', 'LCSceneBuilder', 'LCSeed', 'LCSeedJump', 'LCShowText', 'LCSigmaCurve', 'LCSigmaResample', 'LCSkinBeauty', 'LCSkinUpscale', 'LCSlider', 'LCSplitSigmaScheduler', 'LCSplitSigmasAdvanced', 'LCStop', 'LCStyleSelector', 'LCSubject', 'LCSubjectArray', 'LCTextOverlay', 'LCTextRemove', 'LCTextReplace', 'LCToneMatch', 'LCVRAMCacheClear', 'LCVibrance', 'LCVignette', 'LCWatermark', 'LCWidgetToString', 'LCWildcard']\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [krea2edit] nodes v1.2.5 loaded\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m \n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | Import times for custom nodes:\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/websocket_image_save.py\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-krea2edit\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-lora-tag-hash-metadata\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale\n
2026-09-11 05:34:12.420 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-vslinx-nodes\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-Image-Saver\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/rgthree-comfy\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.1 seconds: /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.5 seconds: /comfyui/custom_nodes/ComfyUI-RMBG\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m \n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade  -> 0001_assets, Initial assets schema\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | Revision ID: 0001_assets\n
2026-09-11 05:34:12.422 | info | 1tftbfouxopgsy | Revises: None\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | Create Date: 2025-12-10 00:00:00\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0001_assets -> 0002_merge_to_asset_references, Merge AssetInfo and AssetCacheState into unified asset_references table.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0002_merge_to_asset_references -> 0003_add_metadata_job_id, Add system_metadata and job_id columns to asset_references.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | Change preview_id FK from assets.id to asset_references.id.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0003_add_metadata_job_id -> 0004_drop_tag_type, Drop the vestigial tags.tag_type column.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0004_drop_tag_type -> 0005_allow_case_sensitive_tags, Allow case-sensitive tag names.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0005_allow_case_sensitive_tags -> 0006_add_loader_path, Add loader_path column to asset_references.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Database upgraded from None to 0006_add_loader_path\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Using RAM pressure cache.\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m Starting server\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | \n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m To see the GUI go to: http://127.0.0.1:8188\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | runpod-worker-comfy - API is reachable\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | runpod-worker-comfy - image(s) upload\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | runpod-worker-comfy - image(s) upload complete\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | [32m[INFO][0m got prompt\n
2026-09-11 05:34:12.423 | warning | 1tftbfouxopgsy | [1m[33m[WARNING][0m invalid prompt: {'type': 'missing_node_type', 'message': "Node 'RMBG (remove headshot background)' not found. The custom node may not be installed.", 'details': "Node ID '#234'", 'extra_info': {'node_id': '234', 'class_type': 'RMBG', 'node_title': 'RMBG (remove headshot background)'}}\n
2026-09-11 05:34:12.423 | info | 1tftbfouxopgsy | Finished.
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Total VRAM 24192 MB, total RAM 1547762 MB\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m pytorch version: 2.11.0+cu128\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Set vram state to: NORMAL_VRAM\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Device: cuda:0 NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb : cudaMallocAsync\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Using async weight offloading with 2 streams\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Enabled pinned memory 1392985.0\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Using pytorch attention\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | CUDA initialization passed: 1 device(s) initialized successfully
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | GPU compute benchmark passed: Matrix multiply completed in 93ms
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | All fitness checks passed. (2284.42ms)
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | Jobs in queue: 1
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | Jobs in progress: 1
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | Started.
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m aimdo: /project/src/control.c:276:INFO:comfy-aimdo inited for GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (VRAM: 24192 MB)\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m DynamicVRAM support detected and enabled\n
2026-09-11 08:47:17.493 | info | 1tftbfouxopgsy | [32m[INFO][0m Python version: 3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m ComfyUI version: 0.34.0\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfy-aimdo version: 0.4.15\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfy-kitchen version: 0.2.31\n
2026-09-11 08:47:17.494 | warning | 1tftbfouxopgsy | [1m[33m[WARNING][0m WARNING: Python 3.10 will be EOL on October 31 2026, please consider upgrading to a newer version.\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfyui-frontend-package version: 1.49.6\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfyui-workflow-templates version: 0.11.48\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfyui-embedded-docs version: 0.5.10\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfy-kitchen version: 0.2.31\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m comfy-aimdo version: 0.4.15\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m [Prompt Server] web root: /usr/local/lib/python3.10/dist-packages/comfyui_frontend_package/static\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m Asset seeder disabled\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [32m[INFO][0m [vsLinx] Combo type validation fix applied.\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | \n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [92m[rgthree-comfy] Loaded 48 fantastic nodes. 🎉[0m\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | \n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [33m[rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [0m\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BiRefNet.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BodySegment.py: No module named 'onnxruntime'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Florence2.py: No module named 'matplotlib'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_ImageMaskTools.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_RMBG.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM2Segment.py: No module named 'hydra'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM3Segment.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Warning: diffusers/transformers not available. SDMatte functionality will be limited.\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Segment.py: No module named 'segment_anything'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SegmentV2.py: No module named 'segment_anything'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_YoloV8.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BiRefNet.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BodySegment.py: No module named 'onnxruntime'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Florence2.py: No module named 'matplotlib'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_ImageMaskTools.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_RMBG.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM2Segment.py: No module named 'hydra'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM3Segment.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Warning: diffusers/transformers not available. SDMatte functionality will be limited.\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Segment.py: No module named 'segment_anything'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SegmentV2.py: No module named 'segment_anything'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_YoloV8.py: No module named 'cv2'\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [34m[ComfyUI-RMBG][0m v[93m3.1.0[0m | [93m9 nodes[0m [92mLoaded[0m\n
2026-09-11 08:47:17.494 | info | 1tftbfouxopgsy | [vsLinx] ComfyUI-Impact-Pack not found - skipping the '(Impact-Pack) Interactive Detailer' node. All other vsLinx nodes are unaffected.\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] loading from /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] LUTs → /comfyui/models/luts: 18 new file(s) copied, 0 existing skipped (no overwrite)\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + aspect_ratio: ['AspectRatioSimplifier', 'LCAspectRatioPipeOut', 'LCAspectRatioPipe']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + slider: ['LCSlider']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + anima_regional_canvas: ['AnimaRegionalCanvasInline']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + krea2_regional_canvas: ['Krea2RegionalCanvasInline']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + dynamic_overlay: ['LCDynamicOverlay']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_any_switch: ['LCAnySwitch']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_index_switch: ['LCIndexSwitch']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_custom_combo: ['LCCustomCombo', 'LCCustomComboPanel']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] registered LCComboSelector\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_combo: ['LCComboSelector']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_invert_boolean: ['LCInvertBoolean']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_boolean: ['LCBoolean']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_boolean_switch: ['LCBooleanSwitch', 'LCBooleanFlip', 'LCBooleanValue']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_builder: ['LCSubject', 'LCSubjectArray', 'LCCamera', 'LCLighting', 'LCStyleSelector', 'LCSceneBuilder', 'LCColorPalette', 'LCPromptAssembler']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_join_strings: ['LCJoinStrings']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_show_text: ['LCShowText']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_widget_to_string: ['LCWidgetToString']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_text_replace: ['LCTextReplace']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_text_remove: ['LCTextRemove']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_compare: ['LCIntCompare', 'LCFloatCompare']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_seed_jump: ['LCSeedJump']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_node_snapshot: ['LCNodeSnapshot']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_notify: ['LCNotify']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_civitai_strip: ['LCCivitaiStrip']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_save_text: ['LC123SaveText']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_batch_image: ['LCBatchImage']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_batch_image_comparer: ['LCBatchImageComparer']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_image_split: ['LCImageSplit']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_last_image_holder: ['LCLastImageHolder']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_sampler_configure: ['LCSamplerConfigure', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigurePipe', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_pipe_io: ['LCPipeOut', 'LCPipeEdit', 'LCDetailPipeOut']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_minimax_h3_pipe: ['LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_get_image: ['LCGetImage']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_dimension_resize: ['LCDimensionResize']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_image_mask_resize: ['LCImageMaskResize']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_image_crop: ['LCImageCrop']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_image_grid: ['LCImageGrid']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_image_tools: ['LCImageAdjust', 'LCAutoWhiteBalance', 'LCClarity', 'LCLensFX', 'LCLiftGammaGain', 'LCImageRGB', 'LCFilmGrain', 'LCVibrance', 'LCVignette', 'LCBloom', 'LCImageDenoise', 'LCColorMatch', 'LCToneMatch', 'LCFilmStockBW', 'LCFilmStockColor', 'LCLensProfile', 'LCChromaticAberration', 'LCImageDesaturate']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_skin_beauty: ['LCSkinBeauty']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_skin_upscale: ['LCSkinUpscale']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_phone_look: ['LCPhoneLook']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_apply_lut: ['LCApplyLUT']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_text_overlay: ['LCTextOverlay']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_watermark: ['LCWatermark']\n
2026-09-11 08:47:17.495 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_to_conditioning: ['LCPromptToConditioning', 'LCPromptToConditioningZero']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_split_sigma_scheduler: ['LCSplitSigmaScheduler']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_basic_scheduler: ['LCBasicScheduler']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_split_sigmas_advanced: ['LCSplitSigmasAdvanced']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_prompt_box: ['LCPositive', 'LCNegative']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_vram_cache_clear: ['LCVRAMCacheClear']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_stop: ['LCStop']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_advanced_folder: ['LCAdvancedFolder']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_easy_folder: ['LCEasyFolder']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_save_image: ['LCSaveImage', 'LCSaveImageMetadata']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_any_empty: ['LCAnyEmptyBool', 'LCAnyEmptyInt', 'LCAnyEmptyFloat']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_int_split: ['LCIntSplit']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_change_step_count: ['LCSigmaResample', 'LCChangeStepCount']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] + lc_sigma_curve: ['LCSigmaCurve']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | USDU batch patches applied successfully.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m USDU batch patches applied successfully.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [LC123] total 105 nodes: ['AnimaRegionalCanvasInline', 'AspectRatioSimplifier', 'Krea2RegionalCanvasInline', 'LC123SaveText', 'LCAdvancedFolder', 'LCAnyEmptyBool', 'LCAnyEmptyFloat', 'LCAnyEmptyInt', 'LCAnySwitch', 'LCApplyLUT', 'LCAspectRatioPipe', 'LCAspectRatioPipeOut', 'LCAutoWhiteBalance', 'LCBasicScheduler', 'LCBatchImage', 'LCBatchImageComparer', 'LCBloom', 'LCBoolean', 'LCBooleanFlip', 'LCBooleanSwitch', 'LCBooleanValue', 'LCCamera', 'LCChangeStepCount', 'LCChromaticAberration', 'LCCivitaiStrip', 'LCClarity', 'LCColorMatch', 'LCColorPalette', 'LCComboSelector', 'LCCustomCombo', 'LCCustomComboPanel', 'LCDenoise', 'LCDetailPipeOut', 'LCDimensionResize', 'LCDynamicOverlay', 'LCEasyFolder', 'LCFilmGrain', 'LCFilmStockBW', 'LCFilmStockColor', 'LCFloatCompare', 'LCGetImage', 'LCImageAdjust', 'LCImageCrop', 'LCImageDenoise', 'LCImageDesaturate', 'LCImageGrid', 'LCImageMaskResize', 'LCImageRGB', 'LCImageSplit', 'LCIndexSwitch', 'LCIntCompare', 'LCIntSplit', 'LCInvertBoolean', 'LCJoinStrings', 'LCLastImageHolder', 'LCLensFX', 'LCLensProfile', 'LCLiftGammaGain', 'LCLighting', 'LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut', 'LCNegative', 'LCNodeSnapshot', 'LCNotify', 'LCPhoneLook', 'LCPipeEdit', 'LCPipeOut', 'LCPositive', 'LCPromptAssembler', 'LCPromptToConditioning', 'LCPromptToConditioningZero', 'LCReferenceLatent', 'LCRelight', 'LCSamplerConfigure', 'LCSamplerConfigurePipe', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut', 'LCSaveImage', 'LCSaveImageMetadata', 'LCSceneBuilder', 'LCSeed', 'LCSeedJump', 'LCShowText', 'LCSigmaCurve', 'LCSigmaResample', 'LCSkinBeauty', 'LCSkinUpscale', 'LCSlider', 'LCSplitSigmaScheduler', 'LCSplitSigmasAdvanced', 'LCStop', 'LCStyleSelector', 'LCSubject', 'LCSubjectArray', 'LCTextOverlay', 'LCTextRemove', 'LCTextReplace', 'LCToneMatch', 'LCVRAMCacheClear', 'LCVibrance', 'LCVignette', 'LCWatermark', 'LCWidgetToString', 'LCWildcard']\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [krea2edit] nodes v1.2.5 loaded\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m \n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | Import times for custom nodes:\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/websocket_image_save.py\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-krea2edit\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-lora-tag-hash-metadata\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-vslinx-nodes\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-Image-Saver\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/rgthree-comfy\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.1 seconds: /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m    0.5 seconds: /comfyui/custom_nodes/ComfyUI-RMBG\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m \n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade  -> 0001_assets, Initial assets schema\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | Revision ID: 0001_assets\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | Revises: None\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | Create Date: 2025-12-10 00:00:00\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0001_assets -> 0002_merge_to_asset_references, Merge AssetInfo and AssetCacheState into unified asset_references table.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0002_merge_to_asset_references -> 0003_add_metadata_job_id, Add system_metadata and job_id columns to asset_references.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | Change preview_id FK from assets.id to asset_references.id.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0003_add_metadata_job_id -> 0004_drop_tag_type, Drop the vestigial tags.tag_type column.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0004_drop_tag_type -> 0005_allow_case_sensitive_tags, Allow case-sensitive tag names.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Running upgrade 0005_allow_case_sensitive_tags -> 0006_add_loader_path, Add loader_path column to asset_references.\n
2026-09-11 08:47:17.496 | info | 1tftbfouxopgsy | [32m[INFO][0m Database upgraded from None to 0006_add_loader_path\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | [32m[INFO][0m Using RAM pressure cache.\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | [32m[INFO][0m Starting server\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | \n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | [32m[INFO][0m To see the GUI go to: http://127.0.0.1:8188\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | runpod-worker-comfy - API is reachable\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | runpod-worker-comfy - image(s) upload\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | runpod-worker-comfy - image(s) upload complete\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | [32m[INFO][0m got prompt\n
2026-09-11 08:47:17.497 | warning | 1tftbfouxopgsy | [1m[33m[WARNING][0m invalid prompt: {'type': 'missing_node_type', 'message': "Node 'RMBG (remove headshot background)' not found. The custom node may not be installed.", 'details': "Node ID '#234'", 'extra_info': {'node_id': '234', 'class_type': 'RMBG', 'node_title': 'RMBG (remove headshot background)'}}\n
2026-09-11 08:47:17.497 | info | 1tftbfouxopgsy | Finished.
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | ==========\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | == CUDA ==\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | ==========\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | CUDA Version 12.8.1\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | This container image and its contents are governed by the NVIDIA Deep Learning Container License.\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | By pulling and using the container, you accept the terms and conditions of this license:\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | runpod-worker-comfy: Starting ComfyUI\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | runpod-worker-comfy: Starting RunPod Handler\n
2026-09-11 09:18:55.042 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.schemas\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.tables\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.types\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.constraints\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.defaults\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.autogenerate.comments\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m setup plugin alembic.ext.checkconstraint_byname\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path checkpoints /runpod-volume/models/checkpoints\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path clip /runpod-volume/models/clip\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path clip_vision /runpod-volume/models/clip_vision\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path configs /runpod-volume/models/configs\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path controlnet /runpod-volume/models/controlnet\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path embeddings /runpod-volume/models/embeddings\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path loras /runpod-volume/models/loras\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path upscale_models /runpod-volume/models/upscale_models\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path vae /runpod-volume/models/vae\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path unet /runpod-volume/models/unet\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path text_encoders /runpod-volume/models/text_encoders\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Adding extra search path diffusion_models /runpod-volume/models/diffusion_models\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m \n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | Prestartup times for custom nodes:\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/rgthree-comfy\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m \n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | --- Starting Serverless Worker |  Version 1.12.0 ---\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | Running 7 fitness check(s)...
2026-09-11 09:18:55.043 | warning | gmn2cq13ttodr5 | [1m[33m[WARNING][0m WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.\n
2026-09-11 09:18:55.043 | warning | gmn2cq13ttodr5 | WARNING WARNING WARNING\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | If you are on nvidia 20 series and above it is required that you update your pytorch to cu130 or higher.\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Found comfy_kitchen backend eager: {'available': True, 'disabled': False, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_embedding', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_mxfp8', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'int8_linear', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_mxfp8', 'scaled_mm_nvfp4', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Found comfy_kitchen backend cuda: {'available': True, 'disabled': True, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'convrot_w4a4_linear', 'dequantize_convrot_w4a4_weight', 'dequantize_int8_convrot_weight', 'dequantize_int8_convrot_weight_dtype', 'dequantize_int8_simple', 'dequantize_int8_simple_dtype', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'dequantize_w4a8_int8_weight', 'gemv_awq_w4a16', 'na3d', 'prepare_int4_weight_for_int8_linear', 'quantize_and_rotate_rowwise', 'quantize_convrot_w4a4_weight', 'quantize_int8_convrot_weight', 'quantize_int8_rowwise', 'quantize_int8_tensorwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'quantize_svdquant_w4a4', 'quantize_w4a8_int8_weight', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'rotate_int8_convrot_weight', 'scaled_mm_svdquant_w4a4', 'stochastic_rounding_fp8', 'w4a8_int8_linear']}\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Found comfy_kitchen backend hip: {'available': False, 'disabled': False, 'unavailable_reason': 'PyTorch ROCm/HIP runtime not available', 'capabilities': []}\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Found comfy_kitchen backend triton: {'available': True, 'disabled': True, 'unavailable_reason': None, 'capabilities': ['adaln', 'apply_rope', 'apply_rope1', 'apply_rope1_', 'apply_rope_', 'apply_rope_split_half', 'apply_rope_split_half1', 'apply_rope_split_half1_', 'apply_rope_split_half_', 'dequantize_nvfp4', 'dequantize_per_tensor_fp8', 'int8_linear', 'na3d', 'quantize_and_rotate_rowwise', 'quantize_int8_rowwise', 'quantize_mxfp8', 'quantize_nvfp4', 'quantize_per_tensor_fp8', 'rms_adaln', 'rms_rope', 'rms_rope1', 'rms_rope1_', 'rms_rope_', 'rms_rope_split_half', 'rms_rope_split_half1', 'rms_rope_split_half1_', 'rms_rope_split_half_', 'w4a8_int8_linear']}\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Checkpoint files will always be loaded safely.\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | GPU binary test passed: 1 GPU(s) healthy (CUDA 13.0)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | Memory check passed: 1243.49GB available (of 1511.43GB total)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | Disk space check passed: 44.98GB free (100.0% available)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | Network connectivity passed: Connected to 8.8.8.8 (1ms)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Total VRAM 24192 MB, total RAM 1547703 MB\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m pytorch version: 2.11.0+cu128\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Set vram state to: NORMAL_VRAM\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Device: cuda:0 NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb : cudaMallocAsync\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Using async weight offloading with 2 streams\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Enabled pinned memory 1392932.0\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m Using pytorch attention\n
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | CUDA version check passed: 13.0 (minimum: 11.8)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | CUDA initialization passed: 1 device(s) initialized successfully
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | GPU compute benchmark passed: Matrix multiply completed in 58ms
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | All fitness checks passed. (1702.87ms)
2026-09-11 09:18:55.043 | info | gmn2cq13ttodr5 | [32m[INFO][0m aimdo: /project/src/control.c:276:INFO:comfy-aimdo inited for GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb (VRAM: 24192 MB)\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m DynamicVRAM support detected and enabled\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m Python version: 3.10.12 (main, Aug 31 2026, 10:18:17) [GCC 11.4.0]\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m ComfyUI version: 0.34.0\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfy-aimdo version: 0.4.15\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfy-kitchen version: 0.2.31\n
2026-09-11 09:18:55.477 | warning | gmn2cq13ttodr5 | [1m[33m[WARNING][0m WARNING: Python 3.10 will be EOL on October 31 2026, please consider upgrading to a newer version.\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfyui-frontend-package version: 1.49.6\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfyui-workflow-templates version: 0.11.48\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfyui-embedded-docs version: 0.5.10\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfy-kitchen version: 0.2.31\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m comfy-aimdo version: 0.4.15\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m [Prompt Server] web root: /usr/local/lib/python3.10/dist-packages/comfyui_frontend_package/static\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | [32m[INFO][0m Asset seeder disabled\n
2026-09-11 09:18:55.477 | info | gmn2cq13ttodr5 | Jobs in queue: 1
2026-09-11 09:18:56.653 | info | gmn2cq13ttodr5 | [32m[INFO][0m No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'\n
2026-09-11 09:18:56.653 | info | gmn2cq13ttodr5 | Jobs in progress: 1
2026-09-11 09:18:56.653 | info | gmn2cq13ttodr5 | Started.
2026-09-11 09:18:56.653 | info | gmn2cq13ttodr5 | [32m[INFO][0m [vsLinx] Combo type validation fix applied.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | USDU batch patches applied successfully.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [32m[INFO][0m USDU batch patches applied successfully.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [92m[rgthree-comfy] Loaded 48 exciting nodes. 🎉[0m\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [33m[rgthree-comfy] ComfyUI's new Node 2.0 rendering may be incompatible with some rgthree-comfy nodes and features, breaking some rendering as well as losing the ability to access a node's properties (a vital part of many nodes). It also appears to run MUCH more slowly spiking CPU usage and causing jankiness and unresponsiveness, especially with large workflows. Personally I am not planning to use the new Nodes 2.0 and, unfortunately, am not able to invest the time to investigate and overhaul rgthree-comfy where needed. If you have issues when Nodes 2.0 is enabled, I'd urge you to switch it off as well and join me in hoping ComfyUI is not planning to deprecate the existing, stable canvas rendering all together.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [0m\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BodySegment.py: No module named 'onnxruntime'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Florence2.py: No module named 'matplotlib'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM2Segment.py: No module named 'hydra'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM3Segment.py: No module named 'iopath'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Warning: diffusers/transformers not available. SDMatte functionality will be limited.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Segment.py: No module named 'segment_anything'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SegmentV2.py: No module named 'segment_anything'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_BodySegment.py: No module named 'onnxruntime'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Florence2.py: No module named 'matplotlib'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM2Segment.py: No module named 'hydra'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SAM3Segment.py: No module named 'iopath'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Warning: diffusers/transformers not available. SDMatte functionality will be limited.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_Segment.py: No module named 'segment_anything'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | Error loading /comfyui/custom_nodes/ComfyUI-RMBG/py/AILab_SegmentV2.py: No module named 'segment_anything'\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [34m[ComfyUI-RMBG][0m v[93m3.1.0[0m | [93m37 nodes[0m [92mLoaded[0m\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [vsLinx] ComfyUI-Impact-Pack not found - skipping the '(Impact-Pack) Interactive Detailer' node. All other vsLinx nodes are unaffected.\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] loading from /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] LUTs → /comfyui/models/luts: 18 new file(s) copied, 0 existing skipped (no overwrite)\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + aspect_ratio: ['AspectRatioSimplifier', 'LCAspectRatioPipeOut', 'LCAspectRatioPipe']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + slider: ['LCSlider']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + anima_regional_canvas: ['AnimaRegionalCanvasInline']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + krea2_regional_canvas: ['Krea2RegionalCanvasInline']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + dynamic_overlay: ['LCDynamicOverlay']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_any_switch: ['LCAnySwitch']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_index_switch: ['LCIndexSwitch']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_custom_combo: ['LCCustomCombo', 'LCCustomComboPanel']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] registered LCComboSelector\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_combo: ['LCComboSelector']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_invert_boolean: ['LCInvertBoolean']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_boolean: ['LCBoolean']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_boolean_switch: ['LCBooleanSwitch', 'LCBooleanFlip', 'LCBooleanValue']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_prompt_builder: ['LCSubject', 'LCSubjectArray', 'LCCamera', 'LCLighting', 'LCStyleSelector', 'LCSceneBuilder', 'LCColorPalette', 'LCPromptAssembler']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_join_strings: ['LCJoinStrings']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_show_text: ['LCShowText']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_widget_to_string: ['LCWidgetToString']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_text_replace: ['LCTextReplace']\n
2026-09-11 09:18:56.754 | info | gmn2cq13ttodr5 | [LC123] + lc_text_remove: ['LCTextRemove']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_compare: ['LCIntCompare', 'LCFloatCompare']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_seed_jump: ['LCSeedJump']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_node_snapshot: ['LCNodeSnapshot']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_notify: ['LCNotify']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_civitai_strip: ['LCCivitaiStrip']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_save_text: ['LC123SaveText']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_batch_image: ['LCBatchImage']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_batch_image_comparer: ['LCBatchImageComparer']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_image_split: ['LCImageSplit']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_last_image_holder: ['LCLastImageHolder']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_sampler_configure: ['LCSamplerConfigure', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigurePipe', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_pipe_io: ['LCPipeOut', 'LCPipeEdit', 'LCDetailPipeOut']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_minimax_h3_pipe: ['LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_get_image: ['LCGetImage']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_dimension_resize: ['LCDimensionResize']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_image_mask_resize: ['LCImageMaskResize']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_image_crop: ['LCImageCrop']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_image_grid: ['LCImageGrid']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_image_tools: ['LCImageAdjust', 'LCAutoWhiteBalance', 'LCClarity', 'LCLensFX', 'LCLiftGammaGain', 'LCImageRGB', 'LCFilmGrain', 'LCVibrance', 'LCVignette', 'LCBloom', 'LCImageDenoise', 'LCColorMatch', 'LCToneMatch', 'LCFilmStockBW', 'LCFilmStockColor', 'LCLensProfile', 'LCChromaticAberration', 'LCImageDesaturate']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_skin_beauty: ['LCSkinBeauty']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_skin_upscale: ['LCSkinUpscale']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_phone_look: ['LCPhoneLook']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_apply_lut: ['LCApplyLUT']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_text_overlay: ['LCTextOverlay']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_watermark: ['LCWatermark']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_prompt_to_conditioning: ['LCPromptToConditioning', 'LCPromptToConditioningZero']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_split_sigma_scheduler: ['LCSplitSigmaScheduler']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_basic_scheduler: ['LCBasicScheduler']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_split_sigmas_advanced: ['LCSplitSigmasAdvanced']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_prompt_box: ['LCPositive', 'LCNegative']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_vram_cache_clear: ['LCVRAMCacheClear']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_stop: ['LCStop']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_advanced_folder: ['LCAdvancedFolder']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_easy_folder: ['LCEasyFolder']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_save_image: ['LCSaveImage', 'LCSaveImageMetadata']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_any_empty: ['LCAnyEmptyBool', 'LCAnyEmptyInt', 'LCAnyEmptyFloat']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_int_split: ['LCIntSplit']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_change_step_count: ['LCSigmaResample', 'LCChangeStepCount']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_sigma_curve: ['LCSigmaCurve']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_pass: ['LCImagePass', 'LCMaskPass']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] + lc_bypass_relay: ['LCBypassRelay']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [LC123] total 108 nodes: ['AnimaRegionalCanvasInline', 'AspectRatioSimplifier', 'Krea2RegionalCanvasInline', 'LC123SaveText', 'LCAdvancedFolder', 'LCAnyEmptyBool', 'LCAnyEmptyFloat', 'LCAnyEmptyInt', 'LCAnySwitch', 'LCApplyLUT', 'LCAspectRatioPipe', 'LCAspectRatioPipeOut', 'LCAutoWhiteBalance', 'LCBasicScheduler', 'LCBatchImage', 'LCBatchImageComparer', 'LCBloom', 'LCBoolean', 'LCBooleanFlip', 'LCBooleanSwitch', 'LCBooleanValue', 'LCBypassRelay', 'LCCamera', 'LCChangeStepCount', 'LCChromaticAberration', 'LCCivitaiStrip', 'LCClarity', 'LCColorMatch', 'LCColorPalette', 'LCComboSelector', 'LCCustomCombo', 'LCCustomComboPanel', 'LCDenoise', 'LCDetailPipeOut', 'LCDimensionResize', 'LCDynamicOverlay', 'LCEasyFolder', 'LCFilmGrain', 'LCFilmStockBW', 'LCFilmStockColor', 'LCFloatCompare', 'LCGetImage', 'LCImageAdjust', 'LCImageCrop', 'LCImageDenoise', 'LCImageDesaturate', 'LCImageGrid', 'LCImageMaskResize', 'LCImagePass', 'LCImageRGB', 'LCImageSplit', 'LCIndexSwitch', 'LCIntCompare', 'LCIntSplit', 'LCInvertBoolean', 'LCJoinStrings', 'LCLastImageHolder', 'LCLensFX', 'LCLensProfile', 'LCLiftGammaGain', 'LCLighting', 'LCMaskPass', 'LCMiniMaxH3Pipe', 'LCMiniMaxH3PipeOut', 'LCNegative', 'LCNodeSnapshot', 'LCNotify', 'LCPhoneLook', 'LCPipeEdit', 'LCPipeOut', 'LCPositive', 'LCPromptAssembler', 'LCPromptToConditioning', 'LCPromptToConditioningZero', 'LCReferenceLatent', 'LCRelight', 'LCSamplerConfigure', 'LCSamplerConfigurePipe', 'LCSamplerConfigurePipeOut', 'LCSamplerConfigureSimple', 'LCSamplerConfigureSimplePipeOut', 'LCSaveImage', 'LCSaveImageMetadata', 'LCSceneBuilder', 'LCSeed', 'LCSeedJump', 'LCShowText', 'LCSigmaCurve', 'LCSigmaResample', 'LCSkinBeauty', 'LCSkinUpscale', 'LCSlider', 'LCSplitSigmaScheduler', 'LCSplitSigmasAdvanced', 'LCStop', 'LCStyleSelector', 'LCSubject', 'LCSubjectArray', 'LCTextOverlay', 'LCTextRemove', 'LCTextReplace', 'LCToneMatch', 'LCVRAMCacheClear', 'LCVibrance', 'LCVignette', 'LCWatermark', 'LCWidgetToString', 'LCWildcard']\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [krea2edit] nodes v1.2.5 loaded\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m \n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | Import times for custom nodes:\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/websocket_image_save.py\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-krea2edit\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/comfyui-lora-tag-hash-metadata\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI_UltimateSDUpscale\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-Image-Saver\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/ComfyUI-vslinx-nodes\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.0 seconds: /comfyui/custom_nodes/rgthree-comfy\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.1 seconds: /comfyui/custom_nodes/ComfyUI_LC123_nodes\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m    0.4 seconds: /comfyui/custom_nodes/ComfyUI-RMBG\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m \n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 09:18:56.755 | info | gmn2cq13ttodr5 | [32m[INFO][0m Context impl SQLiteImpl.\n
2026-09-11 09:18:56.868 | info | gmn2cq13ttodr5 | [32m[INFO][0m Will assume non-transactional DDL.\n
2026-09-11 09:18:56.868 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade  -> 0001_assets, Initial assets schema\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | Revision ID: 0001_assets\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | Revises: None\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | Create Date: 2025-12-10 00:00:00\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade 0001_assets -> 0002_merge_to_asset_references, Merge AssetInfo and AssetCacheState into unified asset_references table.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade 0002_merge_to_asset_references -> 0003_add_metadata_job_id, Add system_metadata and job_id columns to asset_references.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | Change preview_id FK from assets.id to asset_references.id.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade 0003_add_metadata_job_id -> 0004_drop_tag_type, Drop the vestigial tags.tag_type column.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade 0004_drop_tag_type -> 0005_allow_case_sensitive_tags, Allow case-sensitive tag names.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Running upgrade 0005_allow_case_sensitive_tags -> 0006_add_loader_path, Add loader_path column to asset_references.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Database upgraded from None to 0006_add_loader_path\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Using RAM pressure cache.\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m Starting server\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | \n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | [32m[INFO][0m To see the GUI go to: http://127.0.0.1:8188\n
2026-09-11 09:18:56.869 | info | gmn2cq13ttodr5 | runpod-worker-comfy - API is reachable\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | runpod-worker-comfy - image(s) upload\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | runpod-worker-comfy - image(s) upload complete\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | [32m[INFO][0m got prompt\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 229:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m * UNETLoader 130:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m   - Value not in list: unet_name: 'Krea 2/Your models/Animosity_Krea2_V1.0_int8_convrot.safetensors' not in []\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 17:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 215:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 222:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 221:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Failed to validate prompt for output 223:\n
2026-09-11 09:18:57.026 | error | gmn2cq13ttodr5 | [1m[31m[ERROR][0m Output will be ignored\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | runpod-worker-comfy - queued workflow with ID 39f6b4f3-17ad-4a3d-94d1-2b4366b115bd\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | runpod-worker-comfy - wait until image generation is complete\n
2026-09-11 09:18:57.026 | info | gmn2cq13ttodr5 | [32m[INFO][0m [32mPrompt executed in 0.15 seconds[0m\n
2026-09-11 09:18:57.338 | info | gmn2cq13ttodr5 | runpod-worker-comfy - image generation is done\n
2026-09-11 09:18:57.605 | info | gmn2cq13ttodr5 | 
2026-09-11 09:18:57.605 | info | gmn2cq13ttodr5 | runpod-worker-comfy - the image does not exist in the output folder\n
2026-09-11 09:18:57.605 | info | gmn2cq13ttodr5 | Finished.

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
    if [ -n "${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}" ]; then AUTH="Authorization: Bearer ${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}"; fi; \
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
