# Copyright (c) 2018-2021, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import time
import numpy as np
import warnings

from .. import utils
from .. import constants
from .basert_session import BaseRTSession


class ONNXRTSession(BaseRTSession):
    def __init__(self, session_name=constants.SESSION_NAME_ONNXRT, **kwargs):
        super().__init__(session_name=session_name, **kwargs)
        self.kwargs['input_data_layout'] = self.kwargs.get('input_data_layout', constants.NCHW)

    def start(self):
        super().start()

    def import_model(self, calib_data, info_dict=None):
        super().import_model(calib_data)

        print('\n\n\n')
        print("Inside import model, onnxsession")
        
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=True)

        print('\n\n\n')
        print("Done interpreter")
        
        self._get_input_output_details_onnx(self.interpreter)

        print('\n\n\n')
        print("Done _get__get_input_output_details_onnxinp")
        
        
        # provide the calibration data and run the import
        for frame_idx, in_data in enumerate(calib_data):
            print('\n\n\n')
            print("Inside calibration")
            # print("in_data_0 shape:", in_data[0].shape)
            # # print("in_data_1 shape:", in_data[1].shape)
            # print("in_data:", in_data[0][:,:,120:130,120:130])
            # print("in_data_1:", in_data[1])
            # print("indata type:", type(in_data) )
            # print("inddata len:", len(in_data))
            # print("inddata shape:", in_data.shape )
            # in_data = in_data.astype(np.uint8)
            calib_dict = self.get_in_dict(in_data)    

            print("Get calib_dict")
            print("input_normalizer:", self.input_normalizer)
            
            if self.input_normalizer is not None:
                calib_dict, _ = self.input_normalizer(calib_dict, {})
            
            print(f"Normalize input: {self.input_normalizer}")
            
            # model may need additional inputs given in extra_inputs
            if self.kwargs['extra_inputs'] is not None:
                calib_dict.update(self.kwargs['extra_inputs'])
            #
            print(f"add extra input:{self.kwargs['extra_inputs']}")
            for d_info in self.interpreter.get_outputs():
                print(f"output name: {getattr(d_info, 'name')}, shape: {getattr(d_info, 'shape')}, type: {getattr(d_info, 'type')}")
                
            output_keys = [getattr(d_info, 'name') for d_info in self.interpreter.get_outputs()] \
                if self.kwargs['output_details'] is not None else None
                
            print("get output keys")
            
            print(output_keys)
            # print(self.interpreter)
            print("get calib keys")
            # print(calib_dict)
            # print(self.interpreter._providers)
            print('-------------')
            # for k, v in calib_dict.items():
                # print(v)
                # print(k, v.shape, v.dtype)
            print('\n\n')
            
            # run the actual import step
            outputs = self.interpreter.run(output_keys, calib_dict)
            
            print("get output")
            
            self._update_output_details(outputs)
        #

        print("================================ import model =============")
        return info_dict

    def start_infer(self):
        super().start_infer()
        # create the underlying interpreter
        self.interpreter = self._create_interpreter(is_import=False)
        # input_details is needed during inference - get it if it is not given
        self._get_input_output_details_onnx(self.interpreter)
        os.chdir(self.cwd)
        return True

    def get_in_dict(self, in_data):
        if not isinstance(in_data, list) and not isinstance(in_data, dict):
            in_data = utils.as_tuple(in_data)        

        if isinstance(in_data, dict):
            return in_data
        
        return {getattr(d_info, 'name'):d for d_info, d in zip(self.interpreter.get_inputs(),in_data)}
        

    def infer_frame(self, input, info_dict=None):
        super().infer_frame(input, info_dict)

        input_dict = self.get_in_dict(input)

        if self.input_normalizer is not None:
            input_dict, _ = self.input_normalizer(input_dict, {})

        # model needs additional inputs given in extra_inputs
        if self.kwargs['extra_inputs'] is not None:
            input_dict.update(self.kwargs['extra_inputs'])
        #
        # output_details is not mandatory, output_keys can be None
        output_keys = [getattr(d_info, 'name') for d_info in self.interpreter.get_outputs()] \
            if self.kwargs['output_details'] is not None else None
        # run the actual inference
        start_time = time.time()
        outputs = self.interpreter.run(output_keys, input_dict)
        info_dict['session_invoke_time'] = (time.time() - start_time)
        self._update_output_details(outputs)
        return outputs, info_dict

    def set_runtime_option(self, option, value):
        self.kwargs["runtime_options"][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs["runtime_options"].get(option, default)

    def _validate_and_fix_model_shapes(self, model_file):
        """
        ONNX 모델의 shape을 검증하고 필요시 수정
        특히 Transpose 노드의 입력 shape 불일치 문제를 해결
        """
        try:
            import onnx
            import tempfile
            
            # 모델 로드
            model = onnx.load(model_file)
            
            # Shape inference 실행
            try:
                onnx.shape_inference.infer_shapes(model, strict_mode=False)
            except Exception as e:
                warnings.warn(f"Shape inference failed: {e}, continuing without it")
            
            # Transpose 노드의 shape 불일치 문제 검증 및 수정
            modified = False
            for node in model.graph.node:
                if node.op_type == 'Transpose':
                    # Transpose 노드의 perm 속성 확인
                    perm = None
                    for attr in node.attribute:
                        if attr.name == 'perm':
                            perm = list(attr.ints)
                            break
                    
                    if perm is not None:
                        expected_rank = len(perm)
                        
                        # 입력 텐서의 shape 확인
                        input_name = node.input[0]
                        input_shape = None
                        
                        # ValueInfo에서 shape 찾기
                        for value_info in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
                            if value_info.name == input_name:
                                shape = [dim.dim_value if dim.dim_value > 0 else (dim.dim_param if dim.dim_param else -1) 
                                        for dim in value_info.type.tensor_type.shape.dim]
                                input_shape = shape
                                break
                        
                        # Shape inference 결과에서 확인
                        if input_shape is None or any(s == -1 for s in input_shape):
                            # Initializer에서 확인
                            for init in model.graph.initializer:
                                if init.name == input_name:
                                    input_shape = list(init.dims)
                                    break
                        
                        if input_shape is not None:
                            actual_rank = len([s for s in input_shape if s != -1])
                            
                            # Rank 불일치 발견
                            if actual_rank != expected_rank and actual_rank > 0:
                                print(f"⚠️  Shape mismatch detected in Transpose node '{node.name}':")
                                print(f"   Expected rank: {expected_rank} (perm: {perm})")
                                print(f"   Actual rank: {actual_rank} (input shape: {input_shape})")
                                
                                # Reshape 노드를 추가하여 shape 수정 시도
                                # 이는 복잡한 작업이므로 경고만 출력
                                warnings.warn(
                                    f"Transpose node '{node.name}' has shape mismatch. "
                                    f"Expected rank {expected_rank} but got {actual_rank}. "
                                    f"This may cause runtime errors. Consider fixing the model."
                                )
            
            # 모델이 수정되었으면 임시 파일로 저장
            if modified:
                # 임시 파일 생성
                temp_model_file = model_file + '.fixed.onnx'
                onnx.save(model, temp_model_file)
                print(f"✅ Fixed model saved to: {temp_model_file}")
                return temp_model_file
            
            return model_file
            
        except Exception as e:
            warnings.warn(f"Model shape validation failed: {e}, using original model")
            return model_file

    def _create_interpreter(self, is_import):
        # move the import inside the function, so that onnxruntime needs to be installed
        # only if some one wants to use it
        import onnxruntime
        # pass options to pybind
        if is_import:
            self.kwargs["runtime_options"]["import"] = "yes"
        else:
            self.kwargs["runtime_options"]["import"] = "no"
        #
        runtime_options = self.kwargs["runtime_options"]
        sess_options = onnxruntime.SessionOptions()
        
        onnxruntime_graph_optimization_level = self.kwargs["runtime_options"].get('onnxruntime:graph_optimization_level', None)
        if onnxruntime_graph_optimization_level is not None:
            # for transformer models, it is necessary to set graph_optimization_level in session options for onnxruntime
            # to onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL so that TIDL can properly handle the model.
            sess_options.graph_optimization_level = onnxruntime_graph_optimization_level
        
        # suppress warnings
        sess_options.log_severity_level = 3

        # 모델 shape 검증 및 수정 (import 단계에서만)
        model_file = self.kwargs['model_file']
        # if is_import:
        #     print("[모델 shape 검증 중...]")
        #     model_file = self._validate_and_fix_model_shapes(model_file)
        #     print(f"[사용할 모델 파일: {model_file}]")

        if self.kwargs['tidl_offload']:
            ep_list = ['TIDLCompilationProvider', 'CPUExecutionProvider'] if is_import else \
                      ['TIDLExecutionProvider', 'CPUExecutionProvider']
            print("interpreter is created with TIDL provider")
            interpreter = onnxruntime.InferenceSession(model_file, providers=ep_list,
                            provider_options=[runtime_options, {}], sess_options=sess_options)
            print("interpreter is created with TIDL provider")
        else:
            ep_list = ['CPUExecutionProvider']
            print("modelpath")
            print(model_file)
            interpreter = onnxruntime.InferenceSession(model_file, providers=ep_list,
                            provider_options=[{}], sess_options=sess_options)
        #
        return interpreter

    def _set_default_options(self):
        runtime_options = self.kwargs.get("runtime_options", {})
        default_options = {
            "platform": constants.TIDL_PLATFORM,
            "version": constants.TIDL_VERSION_STR,
            "tidl_tools_path": self.kwargs["tidl_tools_path"],
            "artifacts_folder": self.kwargs["artifacts_folder"],
            "tensor_bits": self.kwargs.get("tensor_bits", 8),
            "import": self.kwargs.get("import", 'no'),
            # note: to add advanced options here, start it with 'advanced_options:'
            # example 'advanced_options:pre_batchnorm_fold':1
        }
        default_options.update(runtime_options)
        self.kwargs["runtime_options"] = default_options
