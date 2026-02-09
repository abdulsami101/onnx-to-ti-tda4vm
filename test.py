import pickle
import onnx
import onnxruntime as ort
import numpy as np

with open('output_keys', 'rb') as f:
    output_keys = pickle.load(f)
    
with open('calib_dict.pkl', 'rb') as f:
    calib_dict = pickle.load(f)
    
# model_path = "work_dirs/modelartifacts/AM69A/8bits/kadif-3dod_onnxrt_models_pointPillar_1class_qat_v92_lidar_point_pillars_10k_496x432_qat-p2_onnx/model/lidar_point_pillars_10k_496x432_qat-p2.onnx" 
model_path = "models/pointPillar_1class_float32_v92/lidar_point_pillars_10k_496x432.onnx"
model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# print(calib_dict.keys())
# print(calib_dict['inputNet_IN'].shape)
model_inpt = dict()
for inpt in model.get_inputs():
    model_inpt[inpt.name] = np.random.randn(*inpt.shape).astype(np.float32)
    print(inpt)
    print('-------------')

print(model_inpt.keys())
model_inpt['coors'] = model_inpt['coors'].astype(np.int32) 

outputs = model.run(None, model_inpt)

print(outputs)
