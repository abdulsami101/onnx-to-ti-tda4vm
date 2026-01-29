# ONNX Model Conversion Guide for TI TDA4VM Board

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Structure Setup](#project-structure-setup)
4. [Step-by-Step Conversion Process](#step-by-step-conversion-process)
5. [Deployment to Board](#deployment-to-board)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## 🎯 Overview

This guide will help you convert an ONNX model to TI TIDL artifacts that can run on TDA4VM boards. By the end, you'll have optimized model files ready for deployment.

**What you'll achieve:**
- Convert your ONNX model to TI-optimized format
- Generate quantized artifacts for the board
- Get performance estimates (inference time, memory usage)
- Create deployment-ready package

**Time required:** ~30-60 minutes (first time)

---

## ✅ Prerequisites

### 1. Hardware Requirements
- **Development PC:** Linux (Ubuntu 20.04/22.04 recommended)
- **RAM:** Minimum 8GB (16GB recommended)
- **Disk Space:** At least 10GB free
- **Target Board:** TI TDA4VM (or compatible: AM68A, AM69A, AM67A, AM62A)

### 2. Software Requirements
- **Docker:** Installed and running
  ```bash
  # Check Docker installation
  docker --version
  # Should show: Docker version 20.x or higher
  ```

- **Git:** For cloning the repository
  ```bash
  git --version
  ```

### 3. Your Model Requirements
- ✅ ONNX format model file (`.onnx`)
- ✅ Input dimensions known (e.g., 640×384, 224×224)
- ✅ Model type: Detection, Classification, or Segmentation
- ✅ Calibration images (10-50 images in your dataset format)

---

## 📁 Project Structure Setup

### Step 1: Clone the Repository

```bash
# Navigate to your work directory
cd ~/work

# Clone the edgeai-tensorlab repository
git clone https://github.com/TexasInstruments/edgeai-tensorlab.git
cd edgeai-tensorlab

# Checkout the stable release (r10.1 or later)
git checkout r10.1
```

### Step 2: Build Docker Image

```bash
cd edgeai-benchmark

# Build the Docker image (takes 10-15 minutes)
docker build -t benchmark:v1 -f docker/Dockerfile.benchmark .

# Verify the image
docker images | grep benchmark
# Should show: benchmark   v1   ...
```

### Step 3: Organize Your Files

Create this folder structure:

```
edgeai-benchmark/
├── models/
│   └── detection/              # (or classification/segmentation)
│       ├── your_model.onnx     # Your ONNX model
│       └── your_model.prototxt # Model config (for YOLOX/YOLO)
│
├── dependencies/dataset/detection
│   ├── annotations/
│   │   └── instances_val.json  # COCO format annotations
│   └── images/
│       └── val/                # Calibration images (10-50 images)
│           ├── image_001.jpg
│           ├── image_002.jpg
│           └── ...
│
└── run_compilation.sh          # Compilation script (we'll create this)
```

**Tips:**
- For detection models: Use `models/detection/`
- For classification: Use `models/classification/`
- Images should be representative of your real-world use case
- More calibration images = better quantization accuracy

---

## 🚀 Step-by-Step Conversion Process

### Step 1: Prepare Your Dataset Configuration

#### 1.1 Edit Dataset Registry

Open `edgeai_benchmark/datasets/__init__.py`:

```bash
nano edgeai_benchmark/datasets/__init__.py
```

**Find line ~100** and add your dataset category:

```python
# Add your custom dataset category
DATASET_CATEGORY_YOUR_MODEL = 'your_model_dataset'
```

**Find line ~115** and add your dataset entry:

```python
# Add to the dataset registry
'your_model_dataset': {
    'task_type': 'detection',  # or 'classification', 'segmentation'
    'category': DATASET_CATEGORY_YOUR_MODEL,
    'type': COCODetection,     # or ImageClassification, etc.
    'size': 10,                # Number of calibration images
    'split': 'val'
},
```

**Find line ~380** and add the dataset loader:

```python
# Add dataset loader
if dataset_name == 'your_model_dataset':
    dataset_base = settings.datasets_path.replace('/datasets', '/dataset')
    dataset_path_splits = {
        'val': os.path.join(dataset_base, 'annotations', 'instances_val.json')
    }
    dataset_calib_cfg = dict(
        path=dataset_path_splits['val'],
        split='val'
    )
```

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

---

### Step 2: Configure Your Model Pipeline

#### 2.1 Edit Detection Configuration

Open `configs/detection.py`:

```bash
nano configs/detection.py
```

**Scroll to the end** of the file (around line 410) and add your model configuration:

```python
# Your custom model configuration
'your-model-001': utils.dict_update(common_cfg, {
    'task_type': 'detection',
    'dataset_category': DATASET_CATEGORY_YOUR_MODEL,
    'calibration_dataset': settings.dataset_cache[DATASET_CATEGORY_YOUR_MODEL]['calibration_dataset'],
    'input_dataset': settings.dataset_cache[DATASET_CATEGORY_YOUR_MODEL]['input_dataset'],
    
    # IMPORTANT: Input dimensions (Height, Width)
    # Example for 640×384 model: use (384, 640) - Height first!
    'preprocess': preproc_transforms.get_transform_onnx(
        (384, 640),        # Input size: (Height, Width)
        (384, 640),        # Resize size: (Height, Width)
        reverse_channels=False,  # Set True if model expects RGB instead of BGR
        backend='cv2',
        resize_with_pad=False,
        data_layout=constants.NCHW
    ),
    
    # Session configuration
    'session': onnx_session_type(**sessions.get_onnx_session_cfg(
        settings,
        work_dir=work_dir,
        input_optimization=False
    ),
    runtime_options=settings.runtime_options_onnx_np2(
        det_options=True,
        ext_options={
            'object_detection:meta_arch_type': 6,  # 6 for YOLOX, 3 for YOLO, etc.
            # Add more options if needed:
            # 'advanced_options:output_feature_16bit_names_list': ''
        }
    )),
    
    # Model path
    'model_path': 'models/detection/your_model.onnx',
    
    # Number of output classes
    'model_shortlist': 10,  # Change to your number of classes
}),
```

**Key parameters to customize:**
- `(384, 640)`: Replace with your model's input dimensions (Height, Width)
- `reverse_channels`: Use `True` if your model was trained on RGB images
- `meta_arch_type`: 
  - `6` for YOLOX
  - `3` for YOLOv3
  - `5` for YOLOv5
  - See docs for other architectures
- `model_shortlist`: Your number of output classes

**Save and exit**

---

### Step 3: Create Prototxt File (For YOLO Models)

If you're using YOLOX or YOLO models, create a `.prototxt` file:

```bash
nano models/detection/your_model.prototxt
```

**Example prototxt content:**

```protobuf
name: "your_model_name"

# YOLOX configuration
tidl_yolo {
  # Detection heads (adjust based on your model)
  yolo_param {
    input: "/head/Concat_output_0"    # ONNX output node name
    anchor_width: 8.0
    anchor_height: 8.0
  }
  yolo_param {
    input: "/head/Concat_1_output_0"
    anchor_width: 16.0
    anchor_height: 16.0
  }
  yolo_param {
    input: "/head/Concat_2_output_0"
    anchor_width: 32.0
    anchor_height: 32.0
  }
  
  # Detection parameters
  detection_output_param {
    num_classes: 10              # Your number of classes
    code_type: CODE_TYPE_YOLO_X  # YOLO_X for YOLOX
    confidence_threshold: 0.01   # Adjust as needed
    nms_threshold: 0.45          # Non-maximum suppression threshold
    top_k: 200                   # Maximum detections to keep
  }
  
  # Input dimensions (Width, Height - different from preprocessing!)
  in_width: 640
  in_height: 384
}
```

**How to find output node names:**
```bash
# Use Netron to visualize your ONNX model
# Visit: https://netron.app
# Upload your .onnx file and look at the output layer names
```

**Save and exit**

---

### Step 4: Configure Compilation Settings

Edit `settings_base.yaml`:

```bash
nano settings_base.yaml
```

**Find and modify these sections:**

```yaml
# Target device configuration
target_device: TDA4VM  # or AM68A, AM69A, AM67A, AM62A

# Quantization settings
tensor_bits: 16  # Use 16 for better accuracy (hardware will optimize)

# Calibration settings
calibration_frames: 10        # Number of images to use (1-50)
calibration_iterations: 1     # Usually 1 is enough

# Model selection
model_selection:
  - your-model-001            # Your config name from detection.py

# Dataset selection
dataset_selection:
  - your_model_dataset        # Your dataset name from __init__.py

# Performance options (optional)
# runtime_options:
#   advanced_options:high_resolution_optimization: 1
```

**Save and exit**

---

### Step 5: Create Compilation Script

Create `run_compilation.sh`:

```bash
nano run_compilation.sh
```

**Copy this complete script:**

```bash
#!/bin/bash

# ONNX Model Compilation Script for TI Boards
# This script runs the model compilation inside Docker

echo "======================================"
echo "Starting ONNX Model Compilation"
echo "Target Device: TDA4VM"
echo "======================================"

# Run compilation in Docker
docker run --rm \
  -v $(pwd)/..:/opt/code \
  -e LD_LIBRARY_PATH=/opt/code/edgeai-benchmark/tools/tidl_tools_package/TDA4VM/tidl_tools \
  benchmark:v1 \
  bash --login -c "
    echo 'Installing system dependencies...'
    apt-get update -qq && \
    apt-get install -y libgl1-mesa-glx cmake protobuf-compiler && \
    
    echo 'Installing Python packages...'
    pip install pyyaml pycocotools pillow tqdm colorama onnxsim requests && \
    pip install 'onnx>=1.13.0,<1.15.0' && \
    pip install onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com && \
    pip install 'osrt_model_tools @ git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@11_00_08_00#subdirectory=osrt-model-tools' && \
    
    echo 'Installing TIDL tools (this may take a few minutes)...' && \
    pip install -e ./tools --quiet && \
    download-tidl-tools && \
    
    echo 'Installing edgeai-benchmark...' && \
    pip install -e ./[pc] --quiet && \
    
    echo 'Starting model compilation...' && \
    cd /opt/code/edgeai-benchmark && \
    ./run_benchmarks_pc.sh TDA4VM
  "

echo ""
echo "======================================"
echo "Compilation Complete!"
echo "======================================"
echo "Check results in: work_dirs/modelartifacts/TDA4VM/"
```

**Make it executable:**

```bash
chmod +x run_compilation.sh
```

---

### Step 6: Run the Compilation

```bash
# Make sure you're in edgeai-benchmark directory
cd ~/work/edgeai-tensorlab/edgeai-benchmark

# Run the compilation
sudo ./run_compilation.sh
```

**What to expect:**
1. **Installing dependencies** (~5 minutes first time)
2. **Downloading TIDL tools** (~2 minutes, downloads ~150MB)
3. **Calibration phase** (~10-30 seconds)
4. **Quantization and compilation** (~20-60 seconds)
5. **Success message** with compilation time

**Expected output:**
```
RUNNING': ['your-model-001:import'], 'COMPLETED': []
...
SUCCESS: Benchmark - completed: 1/1
```

---

### Step 7: Verify the Output

Check the generated artifacts:

```bash
# Navigate to output directory
cd work_dirs/modelartifacts/TDA4VM/16bits/

# List your model's artifacts
ls -lh your-model-001_*

# Expected files:
# artifacts/
#   ├── subgraph_0_tidl_net.bin      (15-30 MB) - Quantized model
#   ├── subgraph_0_tidl_io_1.bin     (50-100 KB) - I/O config
#   ├── allowedNode.txt              - Node mapping
#   └── onnxrtMetaData.txt           - Metadata
# model/
#   ├── your_model.onnx              - Original model
#   └── your_model_qparams.prototxt  - Quantization params
# result.yaml                        - Performance metrics
```

**Check performance metrics:**

```bash
cat your-model-001_*/result.yaml
```

**Look for these key values:**
```yaml
result:
  num_subgraphs: 1           # Should be ≥1 (success!)
  perfsim_time_ms: 16.83     # Inference time in milliseconds
  perfsim_ddr_transfer_mb: 20.49  # Memory bandwidth
  perfsim_gmacs: 7.99        # Computational complexity
```

**Calculate FPS:**
```
FPS = 1000 / perfsim_time_ms
Example: 1000 / 16.83 = 59.4 FPS
```

---

## 📦 Deployment to Board

### Step 1: Package the Artifacts

```bash
# Create a compressed archive
cd work_dirs/modelartifacts/TDA4VM/16bits/

tar -czf your_model_artifacts.tar.gz your-model-001_*

# Check the archive size
ls -lh your_model_artifacts.tar.gz
# Should be 15-30 MB typically
```

### Step 2: Transfer to Board

**Option A: Using SCP (if board is on network)**
```bash
# Find your board's IP address (check your board's display or router)
BOARD_IP="192.168.1.100"  # Replace with actual IP

# Transfer the archive
scp your_model_artifacts.tar.gz root@$BOARD_IP:/home/root/

# SSH into the board
ssh root@$BOARD_IP
```

**Option B: Using USB Drive**
```bash
# Copy to USB drive
cp your_model_artifacts.tar.gz /media/usb_drive/

# Then: 
# 1. Unplug USB from PC
# 2. Plug into TDA4VM board
# 3. On board, copy from /media/usb_drive/
```

### Step 3: Extract on Board

On the TDA4VM board:

```bash
# Extract the artifacts
tar -xzf your_model_artifacts.tar.gz

# Navigate to artifacts
cd your-model-001_*/artifacts/

# List files
ls -lh
# You should see:
# - subgraph_0_tidl_net.bin
# - subgraph_0_tidl_io_1.bin
# - allowedNode.txt
```

### Step 4: Run Inference on Board

**Setup environment on board:**
```bash
# Set TIDL library path
export LD_LIBRARY_PATH=/usr/lib/tidl_tools:$LD_LIBRARY_PATH

# Verify TIDL tools are available
ls /usr/lib/tidl_tools/libtidl_onnxrt_EP.so
```

**Run inference (example Python script):**

```python
# save as test_inference.py on the board
import onnxruntime as ort
import numpy as np
from PIL import Image

# Create TIDL session
sess = ort.InferenceSession(
    "../model/your_model.onnx",
    providers=[
        ('TIDLExecutionProvider', {
            'artifacts_folder': './artifacts/',
        }),
        'CPUExecutionProvider'
    ]
)

# Load and preprocess image
img = Image.open("test_image.jpg")
img = img.resize((640, 384))  # Use your model's dimensions
img_array = np.array(img).transpose(2, 0, 1)  # HWC to CHW
img_array = np.expand_dims(img_array, 0).astype(np.float32)

# Run inference
outputs = sess.run(None, {sess.get_inputs()[0].name: img_array})

print(f"Output shape: {outputs[0].shape}")
print(f"Detections: {outputs[0]}")
```

**Run the test:**
```bash
python3 test_inference.py
```

---

## 🔧 Troubleshooting

### Issue 1: Docker Build Fails

**Error:** `Cannot connect to Docker daemon`

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker

# Check status
sudo systemctl status docker

# Add your user to docker group (to avoid sudo)
sudo usermod -aG docker $USER
# Log out and log back in
```

---

### Issue 2: Input Dimension Mismatch

**Error:** `Expected: 384, Got: 640`

**Problem:** Preprocessing dimensions don't match model expectations

**Solution:**
- In `configs/detection.py`, check your `get_transform_onnx()` dimensions
- Format is **(Height, Width)**, not (Width, Height)
- If your model expects 640×384, use `(384, 640)`

**How to verify your model's expected input:**
```bash
# Install onnx
pip install onnx

# Check model
python3 -c "
import onnx
model = onnx.load('models/detection/your_model.onnx')
print(model.graph.input[0])
"
# Look at the shape: [1, 3, H, W]
```

---

### Issue 3: No TIDL Artifacts Generated

**Error:** `num_subgraphs: 0` in result.yaml

**Problem:** TIDL providers not available, model ran on CPU only

**Solution:**

1. **Check run.log for errors:**
```bash
cat work_dirs/.../run.log | grep -i "error\|tidl\|provider"
```

2. **Verify TIDL tools installed:**
```bash
docker run --rm -v $(pwd)/..:/opt/code benchmark:v1 \
  bash -c "ls -la /opt/code/edgeai-benchmark/tools/tidl_tools_package/"
# Should show TDA4VM/ directory
```

3. **Ensure download-tidl-tools ran:**
```bash
# Check your run_compilation.sh includes:
pip install -e ./tools --quiet && download-tidl-tools
```

4. **Verify Python package:**
```bash
docker run --rm -v $(pwd)/..:/opt/code benchmark:v1 \
  bash -c "python3 -c 'import onnxruntime; print(onnxruntime.get_available_providers())'"
# Should include: ['TIDLCompilationProvider', 'TIDLExecutionProvider', ...]
```

---

### Issue 4: Compilation Takes Too Long

**Problem:** Stuck at calibration for >5 minutes

**Solutions:**

1. **Reduce calibration images:**
```yaml
# In settings_base.yaml
calibration_frames: 5  # Try with fewer images
```

2. **Check dataset path:**
```bash
# Verify images exist
ls dataset/images/val/ | wc -l
# Should match your calibration_frames count
```

3. **Check Docker resources:**
```bash
docker stats
# Ensure container has enough RAM (4GB+ recommended)
```

---

### Issue 5: Model Accuracy Drop on Board

**Problem:** Model works on PC but poor accuracy on board

**Possible causes:**

1. **Quantization effects:**
   - Solution: Increase `calibration_frames` to 20-50
   - Use more diverse calibration images

2. **Preprocessing mismatch:**
   - Verify `reverse_channels` setting
   - Check mean/scale normalization matches training

3. **Input format:**
   - Ensure data_layout is correct (NCHW vs NHWC)
   - Verify image resize method matches training

**Debug steps:**
```python
# Compare PC vs Board outputs
# On PC:
import onnxruntime as ort
sess_pc = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
output_pc = sess_pc.run(None, {sess_pc.get_inputs()[0].name: test_input})

# On Board:
sess_board = ort.InferenceSession("model.onnx", providers=['TIDLExecutionProvider'])
output_board = sess_board.run(None, {sess_board.get_inputs()[0].name: test_input})

# Compare
import numpy as np
diff = np.abs(output_pc[0] - output_board[0])
print(f"Max difference: {diff.max()}")
print(f"Mean difference: {diff.mean()}")
# Small differences (<0.1) are normal due to quantization
```

---

### Issue 6: Unsupported ONNX Operations

**Error:** `Unsupported ONNX opset version` or `Unsupported operator`

**Solution:**

1. **Check ONNX version compatibility:**
```bash
# TIDL supports ONNX opset 11-13
# Convert your model to compatible opset:
python3 -c "
import onnx
from onnx import version_converter

model = onnx.load('your_model.onnx')
model_13 = version_converter.convert_version(model, 13)
onnx.save(model_13, 'your_model_opset13.onnx')
"
```

2. **Simplify the model:**
```bash
pip install onnxsim
python3 -m onnxsim your_model.onnx your_model_simplified.onnx
```

3. **Check supported operators:**
   - See TI documentation for supported ops
   - Unsupported ops will run on CPU (slower but still works)

---

## ⚡ Performance Optimization

### 1. Improve Inference Speed

**Reduce input resolution:**
```python
# In configs/detection.py
# Instead of (384, 640), try:
'preprocess': preproc_transforms.get_transform_onnx(
    (288, 480),  # 25% smaller = ~1.5x faster
    ...
)
```

**Adjust NMS parameters:**
```protobuf
# In your_model.prototxt
detection_output_param {
  confidence_threshold: 0.05  # Higher = fewer detections = faster
  nms_threshold: 0.50         # Higher = more aggressive NMS = faster
  top_k: 100                  # Lower = fewer outputs = faster
}
```

**Enable optimizations:**
```yaml
# In settings_base.yaml
runtime_options:
  advanced_options:high_resolution_optimization: 1
  advanced_options:pre_batchnorm_fold: 1
```

---

### 2. Improve Accuracy

**Use more calibration data:**
```yaml
# In settings_base.yaml
calibration_frames: 50  # More samples = better quantization
calibration_iterations: 10  # More iterations = better convergence
```

**Try mixed precision:**
```yaml
# Keep sensitive layers in 16-bit
tensor_bits: 16
# Add to runtime_options:
advanced_options:output_feature_16bit_names_list: '/head/conv1,/head/conv2'
```

**Improve calibration dataset:**
- Include edge cases (dark, bright, occluded objects)
- Cover all object classes evenly
- Use images similar to deployment scenarios

---

### 3. Reduce Model Size

**Quantize to 8-bit:**
```yaml
# In settings_base.yaml
tensor_bits: 8  # Smaller model, slightly lower accuracy
```

**Prune unused outputs:**
```python
# If your model has multiple outputs but you only need one
import onnx

model = onnx.load('your_model.onnx')
# Keep only the output you need
onnx.utils.extract_model(
    'your_model.onnx',
    'your_model_pruned.onnx',
    input_names=['input'],
    output_names=['detections']  # Only the output you use
)
```

---

### 4. Benchmark Different Configurations

Create a test script:

```bash
#!/bin/bash
# benchmark_configs.sh

CONFIGS=("8bit" "16bit" "optimized")
CALIB_FRAMES=(10 25 50)

for config in "${CONFIGS[@]}"; do
  for frames in "${CALIB_FRAMES[@]}"; do
    echo "Testing: $config with $frames frames"
    
    # Update settings
    sed -i "s/calibration_frames: .*/calibration_frames: $frames/" settings_base.yaml
    
    # Run compilation
    ./run_compilation.sh
    
    # Extract metrics
    cat work_dirs/.../result.yaml | grep "perfsim_time_ms"
  done
done
```

---

## 📊 Performance Metrics Explained

### Understanding result.yaml

```yaml
result:
  num_subgraphs: 1              # Number of TIDL-optimized subgraphs (>0 = good)
  perfsim_time_ms: 16.83        # Estimated inference time on board
  perfsim_ddr_transfer_mb: 20.49  # Memory bandwidth usage
  perfsim_gmacs: 7.99           # Computational complexity
  
session:
  num_tidl_subgraphs: 16        # Internal TIDL optimizations (higher = better)
  tensor_bits: 8                # Actual quantization used
  
input_details:
  - shape: [1, 3, 384, 640]     # Model input shape (N, C, H, W)
    dtype: float32
    
output_details:
  - shape: [1, 5040, 15]        # Model output shape
    dtype: float32
```

**Key indicators:**

| Metric | Good | Needs Improvement |
|--------|------|-------------------|
| num_subgraphs | ≥1 | 0 (CPU only) |
| perfsim_time_ms | <20ms (>50 FPS) | >50ms (<20 FPS) |
| num_tidl_subgraphs | >10 | <5 |
| perfsim_ddr_transfer_mb | <50 MB | >100 MB |

**FPS calculation:**
```
FPS = 1000 / perfsim_time_ms

Examples:
- 10ms → 100 FPS (excellent for real-time)
- 20ms → 50 FPS (good for most applications)
- 50ms → 20 FPS (acceptable for non-critical tasks)
```

---

## 📝 Quick Reference Checklist

### Before Starting:
- [ ] Docker installed and running
- [ ] Repository cloned and Docker image built
- [ ] ONNX model file ready
- [ ] 10-50 calibration images prepared
- [ ] Model input dimensions known
- [ ] Prototxt created (for YOLO models)

### Configuration Files:
- [ ] `edgeai_benchmark/datasets/__init__.py` - Dataset registered
- [ ] `configs/detection.py` - Model pipeline configured
- [ ] `settings_base.yaml` - Compilation settings updated
- [ ] `run_compilation.sh` - Compilation script created

### After Compilation:
- [ ] result.yaml shows num_subgraphs ≥ 1
- [ ] artifacts/ folder contains .bin files
- [ ] Performance metrics are acceptable
- [ ] Artifacts packaged in .tar.gz

### On Board:
- [ ] Artifacts transferred and extracted
- [ ] TIDL tools available on board
- [ ] Test inference runs successfully
- [ ] Accuracy validated with test images

---

## 🎓 Additional Resources

### Official Documentation:
- **TI EdgeAI Documentation:** https://github.com/TexasInstruments/edgeai
- **TIDL Tools Guide:** https://software-dl.ti.com/jacinto7/esd/processor-sdk-linux-edgeai/
- **Model Zoo:** https://github.com/TexasInstruments/edgeai-modelzoo

### Model Visualization:
- **Netron:** https://netron.app (visualize ONNX models)
- **ONNX Runtime:** https://onnxruntime.ai/docs/

### Community Support:
- **TI E2E Forums:** https://e2e.ti.com/
- **GitHub Issues:** https://github.com/TexasInstruments/edgeai-tensorlab/issues

---

## ❓ FAQ

**Q: How long does compilation take?**
A: First time ~10-15 minutes (downloading tools), subsequent runs ~1-2 minutes

**Q: Can I use this for custom architectures?**
A: Yes, but you may need to adjust `meta_arch_type` and runtime options. Check docs for supported architectures.

**Q: What if my model is too large?**
A: Try model pruning, reduce input resolution, or use 8-bit quantization instead of 16-bit.

**Q: Do I need to recompile for different boards?**
A: Yes, each board type (TDA4VM, AM68A, etc.) needs separate compilation.

**Q: Can I run multiple models simultaneously?**
A: Yes, but you'll need to compile each model separately and manage memory allocation carefully.

**Q: What's the minimum accuracy loss from quantization?**
A: Typically 1-5% with proper calibration. Use more diverse calibration images to minimize loss.

---

## 🎉 Success Criteria

You've successfully completed the conversion when:

1. ✅ Compilation completes without errors
2. ✅ `num_subgraphs: 1` or higher in result.yaml
3. ✅ Artifacts folder contains .bin files (15-30 MB total)
4. ✅ Performance metrics meet your requirements
5. ✅ Model runs on board with acceptable accuracy

**Congratulations!** Your model is now optimized for the TI TDA4VM board! 🚀

---

## 📞 Need Help?

If you're stuck:

1. **Check the Troubleshooting section** above
2. **Review run.log** in your work_dirs output folder
3. **Verify all file paths** are correct
4. **Ask on TI E2E forums** with your error logs
5. **Check GitHub issues** for similar problems

**Remember:** Most issues are related to:
- Incorrect input dimensions
- Missing TIDL tools installation
- Dataset path or format problems

Happy deploying! 🎯
