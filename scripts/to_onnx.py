"""An example script showing how to export a the example model to ONNX format."""

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import numpy as np
import onnx
import onnxruntime
import torch as T
from torch import nn


# Load the model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 6)

    def forward(self, jets, lep, met, misc):
        o2 = T.sum(jets) + T.sum(lep) + T.sum(met) + T.sum(misc)
        o2 = o2.unsqueeze(0)
        o1 = self.linear(o2)
        return o1, o2


# Define the dummy inputs
jets = T.randn(10, 6)
lep = T.randn(4, 6)
met = T.randn(3)
misc = T.randn(3)

# Create the dummy model
model = DummyModel()
out1, out2 = model(jets, lep, met, misc)

T.onnx.export(
    model=model,
    args=(jets, lep, met, misc),
    f="flow.onnx",
    export_params=True,
    verbose=True,
    opset_version=16,
    dynamic_axes={"jets": {0: "num_jets"}, "lep": {0: "num_lep"}},
    input_names=["jets", "lep", "met", "misc"],
    output_names=["neutrinos", "log_probs"],
)

onnx_model = onnx.load("flow.onnx")
onnx.checker.check_model(onnx_model)

# Print the model graph
print(onnx.helper.printable_graph(onnx_model.graph))

# Run the model using ONNX runtime
ort_session = onnxruntime.InferenceSession("flow.onnx")
ort_inputs = {
    "jets": jets.numpy(),
    "lep": lep.numpy(),
    "met": met.numpy(),
    "misc": misc.numpy(),
}
ort_outs = ort_session.run(None, ort_inputs)

input_names = [i.name for i in ort_session.get_inputs()]
output_names = [o.name for o in ort_session.get_outputs()]

# Check that the outputs are the same
torch_outs = out1.detach().numpy()
assert np.allclose(ort_outs[0], torch_outs, rtol=1e-3, atol=1e-3)

# Test it works with a different number of jets and leptons
ort_inputs["jets"] = T.randn(5, 6).numpy()
ort_inputs["lep"] = T.randn(3, 6).numpy()
ort_outs = ort_session.run(None, ort_inputs)

# Create dummy inputs using the model's input dimensions
# inputs = tuple(T.randn(1, 1, v) for v in model.input_dimensions.values())
# input_names = list(model.input_dimensions.keys())
# output_names = [*list(model.target_dimensions.keys()), "log_probs"]

# # Sanitize the model for ONNX export
# prepare_for_onnx(model, inputs, "forward")

# T.onnx.export(
#     model=model,
#     args=inputs,
#     f="flow.onnx",
#     export_params=True,
#     verbose=True,
#     input_names=input_names,
#     output_names=output_names,
#     opset_version=16,
# )

# onnx_model = onnx.load("flow.onnx")
# onnx.checker.check_model(onnx_model)

# # import onnxruntime
# ort_session = onnxruntime.InferenceSession("flow.onnx")
# ort_inputs = {ort_session.get_inputs()[0].name: c.numpy()}
# ort_outs = ort_session.run(["samples"], ort_inputs)
