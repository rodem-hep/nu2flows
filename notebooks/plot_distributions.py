import h5py
import numpy as np
import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from mltools.mltools.plotting import plot_multi_hists

file_dir = "/srv/beegfs/scratch/groups/dpnc/atlas/ttbar_vflows/data/rel24_240209/hdf5/"
file_name = "merged.h5"
file_path = file_dir + file_name

n_events = None
group_name = "odd"

data = {}
with h5py.File(file_path, "r") as f:
    data["jet"] = f["even"]["jet"][:n_events]
    data["lep"] = f["even"]["lep"][:n_events]
    data["met"] = f["even"]["met"][:n_events]
    data["nu"] = f["even"]["neutrinos"][:n_events]

# Safety clips: make sure all kinematics are < 1 TeV
# Kinematics are the first 3 columns of each array except met which is the first 2
# Clip sumet to 5 TeV
for k, v in data.items():
    if k == "met":
        data[k][..., :2] = np.clip(v[..., :2], -1e3, 1e3)
        data[k][..., 2] = np.clip(v[..., 2], 0, 5e3)
    else:
        data[k][..., :3] = np.clip(v[..., :3], -1e3, 1e3)

# Jets need to be padded so create a mask
mask = data["jet"][..., 3] > 0.01
data["jet"] = data["jet"][mask]

# Plot the distributions
np.set_printoptions(formatter={"float": lambda x: f"{x:0.2e}"})
for k, v in data.items():
    print(f"Array: {k}")
    print(f" - shape: {v.shape}")
    print(f" - nans: {np.isnan(v).sum()}")
    print(f" - infs: {np.isinf(v).sum()}")
    print(f" - min: {np.min(v, axis=tuple(range(v.ndim - 1)))}")
    print(f" - max: {np.max(v, axis=tuple(range(v.ndim - 1)))}")

    flat_v = v.reshape(v.shape[0], -1)
    plot_multi_hists(
        data_list=flat_v,
        data_labels=k,
        col_labels=list(range(flat_v.shape[-1])),
        path=f"plots/{k}_distributions2.png",
        bins=51,
        logy=True,
    )
