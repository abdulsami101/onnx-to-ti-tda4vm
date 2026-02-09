# Complete Guide: Converting ONNX Model to TI Board (ICMS Detection Model)

This guide walks you through the complete process of converting your custom ONNX detection model (`icms-detect-001`) for use on TI embedded boards (like TDA4VM). The process includes model compilation, quantization, and packaging.

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Prepare Your Dataset](#step-1-prepare-your-dataset)
3. [Step 2: Configure Your Model](#step-2-configure-your-model)
4. [Step 3: Run Model Compilation](#step-3-run-model-compilation)
5. [Step 4: Package for Deployment](#step-4-package-for-deployment)
6. [Step 5: Deploy to TI Board](#step-5-deploy-to-ti-board)

---

## Prerequisites

### System Requirements
- **Python**: 3.10 
- **TIDL Tools**: Version 9.2 (matching the version you need for compilation)
- **Docker** (optional but recommended for consistent environment)

### Install TIDL Tools

Download and install TIDL Tools version 9.2 using the setup script:

```bash
bash setup_pc.sh
```

This script will:
- Download TIDL Tools 9.2
- Install necessary dependencies
- Configure environment variables

### Setup Development Environment

Use Docker for a pre-configured environment (recommended):

```bash
cd docker/
bash docker_build.sh      # Build the Docker image
bash docker_run.sh        # Run the container
```

Or install locally using conda/pip (if not using Docker):

```bash
conda activate /home/deltax/work/onnx_model_conversion/edgeai_benchmark/.conda
pip install -r requirements_pc.txt
```

---

## Step 1: Prepare Your Dataset

Your dataset is used for **calibration** - quantizing the model to 16-bit or 8-bit integer format.

### Directory Structure

Create this structure in `dependencies/datasets/`:

```
dependencies/datasets/icms_det/
├── images/                 # All your input images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/            # COCO format annotations
    └── instances_val.json
```

### Annotations Format

Your `instances_val.json` must follow **COCO format**:

```json
{
  "images": [
    {"id": 1, "file_name": "image1.jpg", "height": 384, "width": 640},
    {"id": 2, "file_name": "image2.jpg", "height": 384, "width": 640}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "class_name_1"},
    {"id": 2, "name": "class_name_2"},
    ...
    {"id": 10, "name": "class_name_10"}
  ]
}
```

**Key Points:**
- 10 classes in your dataset
- Image dimensions: 384×640 (matches your model's input)
- At least 1 calibration image (recommended: 50+ for better quantization)

---

## Step 2: Configure Your Model

### 2a. Add Dataset Configuration

Edit `edgeai_benchmark/datasets/__init__.py`:

```python
# Add at the beginning with other dataset categories
DATASET_CATEGORY_ICMS_DET = 'icms_det'  # Your custom dataset

# Add to dataset_info dictionary
'icms_det': {
    'task_type': 'detection',
    'category': DATASET_CATEGORY_ICMS_DET,
    'type': COCODetection,
    'size': 10000,
    'split': 'images'
},

# In the dataset loading section, add this block
if check_dataset_load(settings, DATASET_CATEGORY_ICMS_DET) and \
   (DATASET_CATEGORY_ICMS_DET in dataset_list):
    
    print(utils.log_color("\nINFO", f"loading dataset", 
          f"category:{DATASET_CATEGORY_ICMS_DET}"))

    icms_det_calib_cfg = dict(
        num_classes=10,
        image_dir=f'{settings.datasets_path}/icms_det/images',
        annotation_file=f'{settings.datasets_path}/icms_det/annotations/instances_val.json',
        shuffle=True,
        num_frames=settings.calibration_frames,
        name=DATASET_CATEGORY_ICMS_DET
    )
    
    icms_det_val_cfg = dict(
        num_classes=10,
        image_dir=f'{settings.datasets_path}/icms_det/images',
        annotation_file=f'{settings.datasets_path}/icms_det/annotations/instances_val.json',
        shuffle=False,
        num_frames=settings.num_frames,
        name=DATASET_CATEGORY_ICMS_DET
    )
    
    dataset_cache[DATASET_CATEGORY_ICMS_DET]['calibration_dataset'] = \
        COCODetection(**icms_det_calib_cfg, download=False)
    
    dataset_cache[DATASET_CATEGORY_ICMS_DET]['input_dataset'] = \
        COCODetection(**icms_det_val_cfg, download=False)
```

### 2b. Add Model Configuration

Edit `configs/detection.py` and add your model config in the `get_configs()` function:

```python
'icms-detect-001': utils.dict_update(
    {
        'task_type': 'detection',
        'dataset_category': datasets.DATASET_CATEGORY_ICMS_DET,
        'calibration_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ICMS_DET]['calibration_dataset'],
        'input_dataset': settings.dataset_cache[datasets.DATASET_CATEGORY_ICMS_DET]['input_dataset'],
    },
    preprocess=preproc_transforms.get_transform_onnx(
        resize=(384, 640),
        crop=(384, 640),
        reverse_channels=False,
        data_layout=constants.NCHW,
        backend='cv2',
        interpolation=cv2.INTER_LINEAR,
        resize_with_pad=[True, "corner"],
        add_flip_image=False,
        pad_color=[114, 114, 114]
    ),
    session=onnx_session_type(
        **sessions.get_common_session_cfg(settings, work_dir=work_dir),
        runtime_options=settings.runtime_options_onnx_np2(
            det_options=True,
            ext_options={
                'object_detection:meta_arch_type': 6,
                'object_detection:meta_layers_names_list': 'models/detection/icms1/w_sami.prototxt'
            },
            fast_calibration=True
        ),
        model_path=f'models/detection/icms1/w_sami.onnx'
    ),
    postprocess=postproc_transforms.get_transform_detection_yolov5_onnx(
        squeeze_axis=None,
        normalized_detections=False,
        resize_with_pad=True,
        formatter=postprocess.DetectionBoxSL2BoxLS()
    ),
    metric=dict(label_offset_pred=datasets.label_offset_0to1(num_classes=10)),
    model_info=dict(metric_reference={'accuracy_ap[.5]%': 96.00}, model_shortlist=10)
),
```

**Key Configuration Breakdown:**
- `task_type`: 'detection' - Your model type
- `dataset_category`: ICMS_DET - Points to your dataset
- `preprocess`: Image preprocessing (resize, normalize, etc.)
  - Input size: 384×640
  - Padding color: [114, 114, 114] (typical for YOLO)
- `session`: ONNX runtime configuration
  - `meta_arch_type: 6` - YOLO architecture
  - Calibration enabled for quantization
- `postprocess`: Detection output processing (bounding box formatting)
- `model_info`: Performance reference metrics

### 2c. Update Settings

Edit `settings_base.yaml`:

```yaml
# Target device for compilation
target_device: TDA4VM

# Quantization precision (8 or 16 bits)
tensor_bits: 16

# Frames for inference testing
num_frames: 5

# Frames for calibration (post-training quantization)
calibration_frames: 1

# Optional - runtime options for advanced optimization
# runtime_options:
#   accuracy_level: 1

# Path to your datasets
datasets_path: './dependencies/datasets'

# Model selection - only compile your model
model_selection: ['icms-detect-001']

# Dataset selection
dataset_selection: ['icms_det']

# Task selection
task_selection: ['detection']

# Detection settings
detection_threshold: 0.3
detection_top_k: 200

# Enable TIDL optimization
tidl_offload: True
input_optimization: True

# Save output visualizations
save_output: True
write_results: True
```

---

## Step 3: Run Model Compilation

### What Happens During Compilation

1. **Loading**: Model is loaded (ONNX format)
2. **Preprocessing Validation**: Preprocessing pipeline is tested
3. **Calibration**: Model is calibrated using your dataset images
4. **Quantization**: Model is quantized to 16-bit (or 8-bit) integers
5. **Compilation**: Optimized for TDA4VM processor
6. **Output**: Compiled artifacts saved to `work_dirs/modelartifacts/TDA4VM/`

### Run Compilation on PC

```bash
# Activate your environment (if not in Docker)
conda activate /home/deltax/work/onnx_model_conversion/edgeai_benchmark/.conda

# Run benchmarks (this compiles and tests your model)
./run_benchmarks_pc.sh TDA4VM
```

**What This Script Does:**
- Loads your model configuration from `settings_base.yaml`
- Selects only `icms-detect-001` for processing
- Uses `icms_det` dataset for calibration
- Compiles for `TDA4VM` target device
- Saves compiled artifacts to `work_dirs/modelartifacts/TDA4VM/`
- Runs inference to validate accuracy

### Monitor Compilation Progress

Check the output for:
```
INFO: Compiling model: icms-detect-001
INFO: Using dataset: icms_det
INFO: Calibration in progress...
INFO: Quantization complete
INFO: Model compiled successfully
```

**Troubleshooting:**
- If compilation fails, check dataset path and COCO JSON format
- Ensure TIDL Tools 9.2 is properly installed
- Verify image paths exist in `dependencies/datasets/icms_det/images/`

---

## Step 4: Package for Deployment

After successful compilation, package the model artifacts for deployment to your TI board.

### Run Packaging Script

```bash
./run_package_artifacts_for_evm.sh
```

**What This Does:**
- Collects compiled model artifacts
- Organizes them in deployment-ready structure
- Packages everything to `work_dirs/modelpackage/TDA4VM/`

### Output Structure

After packaging, you'll have:

```
work_dirs/modelpackage/TDA4VM/
├── icms-detect-001/
│   ├── model.so              # Compiled model binary
│   ├── param.yaml            # Model parameters
│   ├── model.onnx            # Original model (reference)
│   └── artifacts/            # All compilation artifacts
```

---

## Step 5: Deploy to TI Board

### Transfer Files to Board

From your PC, transfer the packaged model to your TI board:

```bash
# From PC - copy to board
scp -r work_dirs/modelpackage/TDA4VM/icms-detect-001/ \
    user@<board_ip>:/path/to/deployment/
```

### Run on Board

On the TI board, use the compiled model for inference:

```bash
# On the board
python3 inference_script.py \
    --model models/icms-detect-001/model.so \
    --input image.jpg
```

See [usage_evm.md](./docs/usage_evm.md) for detailed EVM deployment instructions.

---

## Complete Workflow Summary

### Quick Reference

```bash
# 1. Setup environment
bash setup_pc.sh                    # Install TIDL Tools 9.2
conda activate ./.conda            # Activate environment

# 2. Prepare dataset
# Place images in: dependencies/datasets/icms_det/images/
# Place annotations in: dependencies/datasets/icms_det/annotations/instances_val.json

# 3. Configure model
# Edit: edgeai_benchmark/datasets/__init__.py (add dataset config)
# Edit: configs/detection.py (add model config)
# Edit: settings_base.yaml (set model/dataset selection)

# 4. Compile model
./run_benchmarks_pc.sh TDA4VM       # Compiles & validates model

# 5. Package for deployment
./run_package_artifacts_for_evm.sh  # Packages compiled artifacts

# 6. Deploy to board
scp -r work_dirs/modelpackage/TDA4VM/icms-detect-001/ \
    user@board_ip:/deployment/path/
```

---

## Configuration Reference

### Model Configuration Parameters (detection.py)

| Parameter | Description | Your Value |
|-----------|-------------|-----------|
| `task_type` | Model task | `detection` |
| `dataset_category` | Dataset to use | `DATASET_CATEGORY_ICMS_DET` |
| `resize` | Input dimensions | `(384, 640)` |
| `meta_arch_type` | Detection architecture | `6` (YOLO) |
| `num_classes` | Number of classes | `10` |
| `tensor_bits` | Quantization (8 or 16) | `16` |
| `calibration_frames` | Calibration samples | `1-50` |

### Settings Configuration (settings_base.yaml)

| Parameter | Description | Your Value |
|-----------|-------------|-----------|
| `target_device` | Compile target | `TDA4VM` |
| `tensor_bits` | Quantization precision | `16` |
| `num_frames` | Test frames | `5` |
| `calibration_frames` | Calibration frames | `1` |
| `model_selection` | Models to compile | `['icms-detect-001']` |
| `dataset_selection` | Datasets to use | `['icms_det']` |
| `task_selection` | Tasks to run | `['detection']` |

---

## Important Notes

### Preprocessing
- Your model expects images of **384×640**
- Images are resized with padding (corner-aligned)
- Padding color is **[114, 114, 114]** (typical YOLO standard)
- No channel reversal needed (input is BGR)

### Quantization
- **16-bit quantization** is safer for custom models
- Reduces model size by ~50% vs float32
- Minimal accuracy loss
- Use **8-bit only** if you need higher compression and can tolerate ~1-2% accuracy drop

### Calibration
- Minimum 1 image (but **50+ recommended** for better results)
- Use diverse images covering all scenarios
- Must be same size/aspect ratio as training data

### COCO Format Requirements
- All images must be listed in `images` array
- All bounding boxes must be in COCO format: `[x, y, width, height]`
- All 10 classes must be defined in `categories`
- `area = width * height` for each annotation

---

## Useful Links

- [TI EdgeAI GitHub](https://github.com/TexasInstruments/edgeai)
- [TIDL Tools Documentation](https://www.ti.com/edgeai)
- [Model Zoo](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-modelzoo)
- [Custom Models Guide](./docs/custom_models.md)
- [EVM Deployment Guide](./docs/usage_evm.md)

---

## Common Issues & Solutions

### Issue: "Dataset not found"
**Solution**: Verify `dependencies/datasets/icms_det/images/` exists with images

### Issue: "COCO JSON format error"
**Solution**: Validate JSON syntax and ensure all images are listed in `images` array

### Issue: "TIDL Tools not found"
**Solution**: Run `bash setup_pc.sh` again to reinstall TIDL Tools 9.2

### Issue: "Calibration failed"
**Solution**: Check calibration images are accessible and in correct format

### Issue: "Memory error during compilation"
**Solution**: Reduce `calibration_frames` or use 8-bit quantization

---

**Author:** Sami (Well Sami)  
**Last Updated:** February 2026  
**TIDL Version:** 9.2  
**Target Device:** TDA4VM
