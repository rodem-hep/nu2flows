"""An example script showing how to export a the example model to ONNX format."""

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

import onnx
import torch as T

from mltools.mltools.flows import prepare_for_onnx
from src.models.nuflows import NuFlows

# Load the model
checkpoint_path = root / "models/example/checkpoints/last-v2.ckpt"
model = NuFlows.load_from_checkpoint(checkpoint_path, map_location="cpu")

# Create dummy inputs using the model's input dimensions
inputs = tuple(T.randn(1, 1, v) for v in model.input_dimensions.values())
input_names = list(model.input_dimensions.keys())
output_names = list(model.target_dimensions.keys()) + ["log_probs"]

# Sanitize the model for ONNX export
prepare_for_onnx(model, inputs, "forward")

T.onnx.export(
    model=model,
    args=inputs,
    f="flow.onnx",
    export_params=True,
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    opset_version=16,
)

onnx_model = onnx.load("flow.onnx")
onnx.checker.check_model(onnx_model)

# import onnxruntime
# ort_session = onnxruntime.InferenceSession("flow.onnx")
# ort_inputs = {ort_session.get_inputs()[0].name: c.numpy()}
# ort_outs = ort_session.run(["samples"], ort_inputs)
