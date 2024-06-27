"""Create a simple plot to check if the neutrino unfolding is working as expected."""

import rootutils

root = rootutils.setup_root(
    search_from="/home/users/l/leighm/nu2flows", pythonpath=True
)

from pathlib import Path

import h5py
import numpy as np
from dotmap import DotMap

from mltools.mltools.plotting import plot_multi_hists
from src.datamodules.physics import Mom4Vec
from src.utils import read_geant4_file

# Paths to the relevant files
data_file = "/srv/beegfs/scratch/groups/dpnc/atlas/ttbar_vflows/data/nonvirtual_mc16d_spincorr.h5"  # noqa
model_file = "/home/users/l/leighm/scratch/Saved_Networks/nu2flows_geant4/trained_on_atlas_even/outputs/nonvirtual_mc16d_spincorr.h5"  # noqa

# Define the variables to plot
nuflow = DotMap({
    "name": "nu2flows",
    "label": r"$\nu^2$-Flows",
    "hist_kwargs": {"color": "b"},
})
nutruth = DotMap({
    "name": "truth_nu",
    "label": r"$\nu$-Truth",
    "hist_kwargs": {"color": "grey", "fill": True, "alpha": 0.5},
    "err_kwargs": {"color": "grey", "hatch": "///"},
})


# Load the event data from the file
file_data = read_geant4_file(Path(data_file), "atlas_odd")
nutruth.nu = Mom4Vec(file_data.neutrinos.mom[:, 0])
nutruth.anti_nu = Mom4Vec(file_data.neutrinos.mom[:, 1])

# Load the data from the model file
with h5py.File(model_file, "r") as f:
    nuflow.nu = Mom4Vec(f["neutrino"][:, 0])
    nuflow.anti_nu = Mom4Vec(f["antineutrino"][:, 0])

# Combine the two neutrino types into a single list
neutrino_list = [nuflow, nutruth]

# Plot the neutrino momentum
plot_multi_hists(
    data_list=[n.nu.mom for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels=["px", "py", "pz", "E"],
    bins=51,
    path=root / "plots" / "nu_momentum.png",
    do_norm=True,
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
    scale=3,
)
plot_multi_hists(
    data_list=[n.anti_nu.mom for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels=["px", "py", "pz", "E"],
    bins=51,
    path=root / "plots" / "antinu_momentum.png",
    do_norm=True,
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
    scale=3,
)

# Get the lepton associated with each neutrino
lep_mask = file_data.leptons.oth[:, 0] == 1
anti_lep_mask = file_data.leptons.oth[:, 0] == 0

# If both leptons are selected zero out the second
lep_mask[lep_mask.sum(-1) == 2, 1] = False
anti_lep_mask[anti_lep_mask.sum(-1) == 2, 1] = False

# Work out which events to drop
has_lep = lep_mask.sum(-1) == 1
has_anti_lep = anti_lep_mask.sum(-1) == 1

# Pull out the lepton and anti-lepton
lep = file_data.leptons.mom[has_lep][lep_mask[has_lep]]
anti_lep = file_data.leptons.mom[has_anti_lep][anti_lep_mask[has_anti_lep]]

# Convert to 4 vectors
lep = Mom4Vec(lep)
anti_lep = Mom4Vec(anti_lep)

# For each neutrino definition in the list, create a top candidate from the triplet
for n in neutrino_list:
    n.W1 = n.nu[has_anti_lep] + anti_lep
    n.W2 = n.anti_nu[has_lep] + lep

# Plot the W and top mass
plot_multi_hists(
    data_list=[n.W1.mass for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels="W mass [GeV]",
    bins=np.linspace(0, 250, 100),
    path=root / "plots" / "nu_Wmass.png",
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
    scale=4,
)
plot_multi_hists(
    data_list=[n.W2.mass for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels="W mass [GeV]",
    bins=np.linspace(0, 250, 100),
    path=root / "plots" / "antinu_Wmass.png",
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
    scale=4,
)
