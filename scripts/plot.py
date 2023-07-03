"""
Create a simple plot comparing the test set neutrino kinematics to truth
"""

import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
from dotmap import DotMap

from utils.plotting import plot_multi_hists_2
from src.datamodules.physics import Mom4Vec
from src.utils import read_dilepton_file

# Paths to the relevant files
data_file = "/home/users/l/leighm/scratch/Data/nu2flows/test.h5"
model_file = root / "nu2flows_models/example_model/outputs/test.h5"
plot_dir = root / "plots"

# Load the event data from the file
file_data = read_dilepton_file(Path(data_file))

# Define the model neutrino as a dict and load the data
nuflow = DotMap(
    {
        "name": "nu2flows",
        "label": r"$\nu^2$-Flows",
        "hist_kwargs": {"color": "b"},
    }
)
with h5py.File(model_file, "r") as f:
    data = f["gen_nu"][:, : 1]
    nuflow.nu = Mom4Vec(data[:, :, 0])
    nuflow.anti_nu = Mom4Vec(data[:, :, 1])

# Define the truth neutrino as a dict and load from the file
nutruth = DotMap(
    {
        "name": "truth_nu",
        "label": r"$\nu$-Truth",
        "nu": Mom4Vec(file_data.neutrinos.mom[:, None, 0, :]),
        "anti_nu": Mom4Vec(file_data.neutrinos.mom[:, None, 1, :]),
        "hist_kwargs": {"color": "grey", "fill": True, "alpha": 0.5},
        "err_kwargs": {"color": "grey", "hatch": "///"},
    }
)

# Combine the two neutrino types into a single list
neutrino_list = [nuflow, nutruth]

# Create the plotting folder
plot_dir.mkdir(parents=True, exist_ok=True)

# Plot the neutrino energy
plot_multi_hists_2(
    data_list=[n.nu.E for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels="Energy [GeV]",
    path=plot_dir / "energy.png",
    bins=np.linspace(0, 600, 100),
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
)

# Create the W bosons for each of the neutrinos
for part in ["nu", "anti_nu"]:

    # Pull out the appropriate lepton from the file
    idx = 1 if part == "nu" else 0
    lep = file_data.leptons[:, idx : idx + 1]

    # Pull out the appropriate bjet from the file
    b_idx = 0 if part == "nu" else 1
    b_loc = file_data.jets_indices == b_idx
    bjet = np.zeros((len(file_data.MET), 1, 4))
    bjet[np.any(b_loc, axis=-1)] = file_data.jets[b_loc].mom[:, None]
    bjet = Mom4Vec(bjet)

    # Go through the neutrino types
    for nu in neutrino_list:

        # Save the w info
        w_name = "W_plus" if part == "nu" else "W_minus"
        nu[w_name] = lep + nu[part]

        # Save the top info
        t_name = "top" if part == "nu" else "anti_top"
        nu[t_name] = nu[w_name] + bjet

# Plot the top mass
plot_multi_hists_2(
    data_list=[n.top.mass[file_data.has_both_bs] for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels="Top mass [GeV]",
    path=plot_dir / "mass.png",
    bins=np.linspace(0, 400, 100),
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
)