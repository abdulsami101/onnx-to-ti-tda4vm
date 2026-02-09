import onnx
import numpy as np
from onnx import numpy_helper

# 1. ONNX 모델 로드
# path = './work_dirs/modelartifacts/TDA4VM/16bits/pha2-gaze-0003_Y_add_mean_scale_onnxrt_pha2_models_gaze_2025_09_24_Y_channel_onnx/model/2025_09_24_Y_channel.onnx'
path = './work_dirs/modelartifacts/TDA4VM/16bits/pha2-gaze-0003_Y_onnxrt_pha2_models_gaze_2025_09_24_Y_channel_onnx/model/2025_09_24_Y_channel.onnx'
model = onnx.load(path)

# 2. Initializer에서 bias와 scale 찾기
bias = None
scale = None
for initializer in model.graph.initializer:
    if initializer.name == "TIDL_preProc_Bias":
        bias = numpy_helper.to_array(initializer)
        print("Add Bias (TIDL_preProc_Bias):", bias)
    if initializer.name == "TIDL_preProc_Scale":
        scale = numpy_helper.to_array(initializer)
        print("Mul Scale (TIDL_preProc_Scale):", scale)

if bias is None or scale is None:
    print("Bias 또는 Scale을 모델에서 찾지 못했습니다.")
    exit(1)

# 3. 임의의 입력 데이터 생성 (예: uint8 이미지)
# 실제 모델 입력 shape 확인 필요 (예: [1, 3, H, W])
input_shape = [1, 3, 224, 224]  # 예시
input_data = np.random.randint(0, 256, size=input_shape).astype(np.uint8)
print("Original Input (inputNet_IN) shape:", input_data.shape)

# 4. Cast 수행 (ONNX Cast처럼 float32로 변환)
input_cast = input_data.astype(np.float32)

# 5. Add 연산
add_out = input_cast + bias.reshape((1, -1, 1, 1))  # 채널 차원 맞추기
print("After Add (TIDL_Scale_In) shape:", add_out.shape)

# 6. Mul 연산
mul_out = add_out * scale.reshape((1, -1, 1, 1))
print("After Mul (final 'input') shape:", mul_out.shape)

# 7. 결과 확인 (최댓값/최솟값)
print("Input min/max:", input_data.min(), input_data.max())
print("After Add min/max:", add_out.min(), add_out.max())
print("After Mul min/max:", mul_out.min(), mul_out.max())
def list_op_types(onnx_path):
    model = onnx.load(onnx_path)
    ops = set(node.op_type for node in model.graph.node)
    print("Operators in model:")
    for op in sorted(ops):
        print(" ", op)
def show_first_nodes(onnx_path, n=10):
    model = onnx.load(onnx_path)
    for i, node in enumerate(model.graph.node[:n]):
        print(f"{i}: {node.op_type}, inputs={node.input}, outputs={node.output}")


list_op_types(path)
# 사용 예시
show_first_nodes(path)