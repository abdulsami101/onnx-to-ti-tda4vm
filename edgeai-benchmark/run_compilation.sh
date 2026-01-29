#!/bin/bash

echo "=================================================="
echo "Running ONNX Model Compilation for TDA4VM"
echo "=================================================="

cd /home/deltax/work/onnx_model_conversion/edgeai-tensorlab/edgeai-benchmark

docker run --rm \
  -v $(pwd)/..:/opt/code \
  --privileged \
  --network host \
  --shm-size 10gb \
  -e LD_LIBRARY_PATH=/opt/code/edgeai-benchmark/tools/tidl_tools_package/TDA4VM/tidl_tools \
  benchmark:v1 \
  bash --login -c "\
    source /opt/.bashrc && \
    cd /opt/code/edgeai-benchmark && \
    echo 'Python version:' && python3 --version && \
    echo 'Installing system dependencies...' && \
    apt-get update -qq && apt-get install -y -qq libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev cmake protobuf-compiler && \
    echo 'Fixing OpenCV installation...' && \
    pip uninstall -y opencv-python opencv-contrib-python && \
    pip install --quiet opencv-python-headless && \
    echo 'Installing required Python packages...' && \
    pip install --quiet pyyaml pycocotools pillow tqdm colorama onnxsim requests && \
    echo 'Installing compatible onnx version...' && \
    pip install --quiet 'onnx>=1.13.0,<1.15.0' && \
    echo 'Installing onnx-graphsurgeon...' && \
    pip install --quiet onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com && \
    echo 'Installing osrt_model_tools...' && \
    pip install --quiet 'osrt_model_tools @ git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@11_00_08_00#subdirectory=osrt-model-tools' && \
    echo 'Installing tools package...' && \
    pip install -e ./tools --quiet && \
    echo 'Downloading TIDL tools (this downloads ~150MB)...' && \
    download-tidl-tools && \
    echo 'Installing edgeai-benchmark package with TIDL support...' && \
    pip install -e ./[pc] --quiet && \
    echo 'Setting library path for TIDL...' && \
    export LD_LIBRARY_PATH=/opt/code/edgeai-benchmark/tools/tidl_tools_package/TDA4VM/tidl_tools:\$LD_LIBRARY_PATH && \
    echo 'Starting compilation...' && \
    ./run_benchmarks_pc.sh TDA4VM"
