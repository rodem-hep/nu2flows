"""Check the contents of a ROOT file."""

import rootutils
from dotmap import DotMap

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import awkward as ak
import numpy as np
import uproot as ur

from mltools.mltools.plotting import plot_multi_correlations, plot_multi_hists
from src.datamodules.physics import Mom4Vec


def awkward3D_to_padded(a: ak.Array, max_len: int | None = None) -> np.ndarray:
    if max_len is None:
        max_len = int(ak.max(ak.num(a)))
    np_arr = ak.pad_none(a, target=max_len, clip=True)
    np_arr = ak.to_numpy(np_arr)
    np_arr = np.array(np_arr)
    return np.nan_to_num(np_arr, 0.0)


# Open the file and get the tree
file_path = Path("/home/users/l/leighm/TopCPToolkit/run/output.root")
file = ur.open(file_path)
tree = file["reco"]

# Load the leptons
keys = ["pt_NOSYS", "eta", "phi", "charge"]
lep = []
for b in ["el", "mu"]:
    x = []
    for k in [f"{b}_{j}" for j in keys]:
        a = awkward3D_to_padded(tree[k].array())
        x.append(a)
        if "pt" in k:
            a /= 1000
    x = np.dstack(x).astype(np.float32)
    lep.append(x)
lep = np.hstack(lep)

# Load the neutrinos
keys = ["pt", "eta", "phi"]
nu = []
for b in ["Ttbar_MC_Wdecay1_from_t", "Ttbar_MC_Wdecay2_from_tbar"]:
    x = []
    for k in [f"{b}_{j}" for j in keys]:
        a = ak.to_numpy(tree[k].array())[..., None]
        if "pt" in k:
            a /= 1000
        x.append(a)
    x = np.dstack(x).astype(np.float32)
    nu.append(x)
nu = np.hstack(nu)

# Load the nuflow outputs
nu_flow = ak.to_numpy(tree["nuflows_nu_out_NOSYS"].array())
nu_flow = np.reshape(nu_flow, (-1, 2, 3))

# Define the variables to plot
nuflow = DotMap({
    "name": "nu2flows",
    "label": r"$\nu^2$-Flows",
    "hist_kwargs": {"color": "b"},
    "nu": Mom4Vec(nu_flow[:, 0]),
    "anti_nu": Mom4Vec(nu_flow[:, 1]),
})
nutruth = DotMap({
    "name": "truth_nu",
    "label": r"$\nu$-Truth",
    "hist_kwargs": {"color": "grey", "fill": True, "alpha": 0.5},
    "err_kwargs": {"color": "grey", "hatch": "///"},
    "nu": Mom4Vec(nu[:, 0], is_cartesian=False),
    "anti_nu": Mom4Vec(nu[:, 1], is_cartesian=False),
})
nutruth.nu.to_cartesian()
nutruth.anti_nu.to_cartesian()

# Combine the two neutrino types into a single list
neutrino_list = [nuflow, nutruth]

# Plot the neutrino momentum
plot_multi_hists(
    data_list=[n.nu.mom for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels=["px", "py", "pz", "E"],
    bins=51,
    path=root / "plots" / "root_nu_momentum.png",
    do_norm=True,
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
)
plot_multi_hists(
    data_list=[n.anti_nu.mom for n in neutrino_list],
    data_labels=[n.label for n in neutrino_list],
    col_labels=["px", "py", "pz", "E"],
    bins=51,
    path=root / "plots" / "root_antinu_momentum.png",
    do_norm=True,
    do_err=True,
    hist_kwargs=[n.hist_kwargs for n in neutrino_list],
    err_kwargs=[n.err_kwargs for n in neutrino_list],
)

# Calculate the W mass per event using the leptons (real)
idxes = np.arange(len(nu))

has_lep = np.any(lep[..., -1] == -1, axis=-1)
lep_mask = np.argmax(lep[..., -1] == -1, axis=-1)
leptons = lep[idxes, lep_mask][has_lep]
leptons = Mom4Vec(leptons, is_cartesian=False, n_mom=3)
leptons.to_cartesian()

has_anti_lep = np.any(lep[..., -1] == 1, axis=-1)
anti_lep_mask = np.argmax(lep[..., -1] == 1, axis=-1)
anti_leptons = lep[idxes, anti_lep_mask][has_anti_lep]
anti_leptons = Mom4Vec(anti_leptons, is_cartesian=False, n_mom=3)
anti_leptons.to_cartesian()

# For each neutrino definition in the list, create a W candidate
for n in neutrino_list:
    n.W1 = n.nu[has_anti_lep] + anti_leptons
    n.W2 = n.anti_nu[has_lep] + leptons

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
)

flow = np.vstack(
    [
        np.hstack([nuflow.nu[has_anti_lep].mom[:, :3], anti_leptons.mom[:, :3]]),
        np.hstack([nuflow.anti_nu[has_lep].mom[:, :3], leptons.mom[:, :3]]),
    ],
)
truth = np.vstack(
    [
        np.hstack([nutruth.nu[has_anti_lep].mom[:, :3], anti_leptons.mom[:, :3]]),
        np.hstack([nutruth.anti_nu[has_lep].mom[:, :3], leptons.mom[:, :3]]),
    ],
)

plot_multi_correlations(
    data_list=[flow, truth],
    data_labels=[n.label for n in neutrino_list],
    col_labels=["nu_px", "nu_py", "nu_pz", "lep_px", "lep_py", "lep_pz"],
    n_bins=31,
    n_kde_points=30,
    levels=4,
    do_err=True,
    do_norm=True,
    path=root / "plots" / "nu_correlations.png",
)
