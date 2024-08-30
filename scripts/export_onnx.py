"""An example script showing how to export a the example model to ONNX format."""

import argparse

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

import logging
from pathlib import Path

import hydra
import onnxruntime
import torch as T

import onnx
from mltools.mltools.hydra_utils import reload_original_config

log = logging.getLogger(__name__)


def atanh_symbolic(g, self):
    return g.op("Atanh", self)


T.onnx.register_custom_op_symbolic("::atanh", atanh_symbolic, 16)


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--project_dir",
        type=str,
        help="the path to the project directory containing the network",
        default="/home/users/l/leighm/scratch/Saved_Networks/nu2flows_geant4/",
    )
    args.add_argument(
        "--network_name",
        type=str,
        help="the name of the network to load",
        default="final_even_long",
    )
    args.add_argument(
        "--output_name",
        type=str,
        help="the name of the output node in the onnx file",
        default="",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        help="the directory to save the output onnx file",
        default="onnx",
    )
    return args.parse_args()


def main() -> None:
    log.info("Parsing arguments")
    args = parse_args()

    log.info("Loading best checkpoint")
    full_path = Path(args.project_dir, args.network_name)
    orig_cfg = reload_original_config(path=full_path, ckpt_flag="*best*")
    model_class = hydra.utils.get_class(orig_cfg.model._target_)
    model = model_class.load_from_checkpoint(orig_cfg.ckpt_path, map_location="cpu")

    # Set model for export
    model.transformer.set_packed(False)  # Flash attention only works on GPUs
    model.disable_masking = True  # Never pass padded sequences to the model
    model.eval()  # Disable dropout

    # Define the dummy inputs based on the model's input dimensions
    input_names, input_shapes = zip(*model.input_dimensions.items())
    dummy_inputs = tuple(T.randn(1, s) for s in input_shapes)

    # Check that the model runs
    model(*dummy_inputs)

    # Export the model to ONNX format
    out_name = args.output_name or args.network_name
    out_path = Path(args.output_dir, out_name + ".onnx")
    output_names = ["outputs", "log_probs"]
    T.onnx.export(
        model=model,
        args=dummy_inputs,
        f=out_path,
        export_params=True,
        verbose=False,
        opset_version=16,  # Version used by ATLAS = 20
        dynamic_axes={k: {0: f"num_{k}"} for k in input_names},
        input_names=list(input_names),
        output_names=output_names,
    )

    # Check that the model is valid
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)

    # log the model graph
    text_name = Path(args.output_dir, args.network_name + ".txt")
    graph = onnx.helper.printable_graph(onnx_model.graph)
    Path(text_name).write_text(graph)

    # Initiate the onnx runtime session
    onnxruntime.set_seed(1)
    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(out_path, sess_options)

    # Print the session input and output names and shapes
    inputs = ort_session.get_inputs()
    outputs = ort_session.get_outputs()
    print("Inputs:")
    for inpt in inputs:
        print(f" - {inpt.name}: {inpt.shape}")
    print("Outputs:")
    for outpt in outputs:
        print(f" - {outpt.name}: {outpt.shape}")

    # Generate new dummy inputs (to check that the trace was data-dependent)
    dummy_inputs = tuple(T.randn(5, s) * 1000 for s in input_shapes)
    torch_outputs = model(*dummy_inputs)[0]

    # Run the model using ONNX runtime
    ort_inputs = {k: v.numpy() for k, v in zip(input_names, dummy_inputs)}
    onnx_outputs = ort_session.run(None, ort_inputs)[0]

    # Check that the outputs are the same
    max_diff = T.max(T.abs(torch_outputs - T.tensor(onnx_outputs)))
    print(f"max diff: {max_diff}")

    # Test it works with a different number of jets and leptons
    ort_inputs["jets"] = T.randn(10, 6).numpy()
    ort_inputs["leptons"] = T.randn(3, 6).numpy()
    ort_session.run(None, ort_inputs)

    print("Hooray! The model is exported and runs with ONNX runtime!")


if __name__ == "__main__":
    main()
