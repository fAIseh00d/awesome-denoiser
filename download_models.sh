#!/bin/bash
set -e

MODEL_DIR="${1:-example_data}"
TORCH_CACHE="/root/.cache/torch/hub/checkpoints"
mkdir -p "$MODEL_DIR"
mkdir -p "$TORCH_CACHE"

echo "=== Downloading models ==="

# Facebook Denoiser DNS models (cached by torch)
echo "[1/9] Downloading Facebook DNS64 model..."
wget --progress=bar:force -O "$TORCH_CACHE/dns64-a7761ff99a7d5bb6.th" \
    "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"

echo "[2/9] Downloading Facebook DNS48 model..."
wget --progress=bar:force -O "$TORCH_CACHE/dns48-11decc9d8e3f0998.th" \
    "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th"

# Mossformer2 model (from ModelScope)
echo "[3/9] Downloading Mossformer2 model..."
wget --progress=bar:force -O "$MODEL_DIR/mossformer2_model.onnx" \
    "https://www.modelscope.cn/models/dengcunqin/speech_mossformer2_noise_reduction_16k/resolve/master/simple_model.onnx"

# DeepFilterNet model
echo "[4/9] Downloading DeepFilterNet model..."
wget --progress=bar:force -O "$MODEL_DIR/denoiser_model.onnx" \
    "https://github.com/yuyun2000/SpeechDenoiser/raw/main/48k/denoiser_model.onnx"

# MDX-NET model (UVR)
echo "[5/9] Downloading MDX-NET model..."
wget --progress=bar:force -O "$MODEL_DIR/UVR-MDX-NET-Inst_HQ_1.onnx" \
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_1.onnx"

# DTLN models
echo "[6/9] Downloading DTLN model 1..."
wget --progress=bar:force -O "$MODEL_DIR/model_1.tflite" \
    "https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_1.tflite"

echo "[7/9] Downloading DTLN model 2..."
wget --progress=bar:force -O "$MODEL_DIR/model_2.tflite" \
    "https://github.com/breizhn/DTLN/raw/refs/heads/master/pretrained_model/model_2.tflite"

# GTCRN model
echo "[8/9] Downloading GTCRN model..."
wget --progress=bar:force -O "$MODEL_DIR/gtcrn.onnx" \
    "https://github.com/Xiaobin-Rong/gtcrn/raw/main/stream/onnx_models/gtcrn.onnx"

# Pre-download ModelScope models using Python
echo "[9/10] Pre-downloading ModelScope FRCRN model..."
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('damo/speech_frcrn_ans_cirm_16k')
print('FRCRN model downloaded successfully')
"

echo "[10/10] Pre-downloading ModelScope ZipEnhancer model..."
python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('iic/speech_zipenhancer_ans_multiloss_16k_base')
print('ZipEnhancer model downloaded successfully')
"

echo ""
echo "=== All models downloaded ==="
ls -lh "$MODEL_DIR"/*.onnx "$MODEL_DIR"/*.tflite 2>/dev/null || true
ls -lh "$TORCH_CACHE"/*.th 2>/dev/null || true