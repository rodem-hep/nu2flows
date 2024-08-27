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
        default="final_even",
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
    output_names, output_shapes = zip(*model.target_dimensions.items())
    output_names += ("log_probs",)
    output_shapes += (1,)

    # Define the dummy inputs based on the model's input dimensions
    dummy_inputs = tuple(T.randn(1, s) for s in input_shapes)

    # Check that the model runs
    _ = model(*dummy_inputs)

    # Export the model to ONNX format
    outname = Path(args.output_dir, args.network_name + ".onnx")
    T.onnx.export(
        model=model,
        args=dummy_inputs,
        f=outname,
        export_params=True,
        verbose=True,
        opset_version=16,
        dynamic_axes={k: {0: f"num_{k}"} for k in input_names},
        input_names=list(input_names),
        output_names=list(output_names),
    )

    # Check that the model is valid
    onnx_model = onnx.load(outname)
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
    ort_session = onnxruntime.InferenceSession(outname, sess_options)

    # Generate new dummy inputs (to check that the trace was data-dependent)
    dummy_inputs = tuple(T.randn(5, s) for s in input_shapes)
    model(*dummy_inputs)

    # Run the model using ONNX runtime
    ort_inputs = {k: v.numpy() for k, v in zip(input_names, dummy_inputs)}
    ort_session.run(None, ort_inputs)

    # Test it works with a different number of jets and leptons
    ort_inputs["jets"] = T.randn(10, 6).numpy()
    ort_inputs["leptons"] = T.randn(3, 6).numpy()
    ort_session.run(None, ort_inputs)

    print("Hooray! The model is exported and runs with ONNX runtime!")


if __name__ == "__main__":
    main()
