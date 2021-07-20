import pickle
import numpy as np
import json
import time
import joblib
import os
import onnxruntime
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global scaler, model, input_name, label_name, inputs_dc, output_dc
    #scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler.pkl')
    scaler_path = Model.get_model_path(model_name='Scaler', version=1)
    scaler = joblib.load(scaler_path)
    
    #model_onnx = os.path.join(os.getenv('AZUREML_MODEL_DIR'), version, 'final_model_svm.onnx')
    model_onnx = Model.get_model_path(model_name='Support-vector-Classifier', version=1)
    model = onnxruntime.InferenceSession(model_onnx, None)
    input_name = model.get_inputs()[0].name
    label_name = model.get_outputs()[0].name
    
    # Variable to monitor model input and output data
    inputs_dc = ModelDataCollector("Support vector classifier model", designation="inputs", feature_names=["feat1", "feat2", "feat3", "feat4", "feat5", "feat6"])
    output_dc = ModelDataCollector("Support vector classifier model", designation="output", feature_names=["weatherprediction"])
# @input_schema("data", NumpyParameterType(np.array([34.927778, 0.24, 7.3899, 83, 16.1000, 1])))
# @output_schema(NumpyParameterType(np.array([0])))

def run(raw_data):
    try:
        #Scaling data
        data = np.array(json.loads(raw_data)['data'])
        data = scaler.transform(data.reshape(1, -1))
        #Infer model
        result = model.run([label_name], {input_name:data.astype(np.float32)})[0]
        result = {'result': np.array(result).tolist()}
        return result
        output_dc.collect(result)
    except Exception as e:
        result = {'result':str(e)}
        return result
        output_dc.collect(result)
