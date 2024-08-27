import h5py
import numpy as np
import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


from mltools.mltools.plotting import plot_multi_hists

file_dir = "/srv/beegfs/scratch/groups/dpnc/atlas/ttbar_vflows/data/rel24_240209/hdf5/"
file_name = "merged.h5"
file_path = file_dir + file_name

n_events = 100_000
group_name = "even"

data = {}
with h5py.File(file_path, "r") as f:
    data["jet"] = f["even"]["jet"][:n_events]
    data["lep"] = f["even"]["lep"][:n_events]
    data["met"] = f["even"]["met"][:n_events]
    data["nu"] = f["even"]["neutrinos"][:n_events]

# Safety clips: make sure all kinematics are < 1 TeV
# Kinematics are the first 3 columns of each array except met which is the first 2
# Clip sumet to 5 TeV
data["jet"][..., :3] = np.clip(data["jet"][..., :3], -1e3, 1e3)
data["lep"][..., :3] = np.clip(data["lep"][..., :3], -1e3, 1e3)
data["nu"][..., :3] = np.clip(data["nu"][..., :3], -1e3, 1e3)
data["met"][..., :2] = np.clip(data["met"][..., :2], -1e3, 1e3)
data["met"][..., 2] = np.clip(data["met"][..., 2], 0, 5e3)

# Jets need to be padded so create a mask
mask = data["jet"][..., 3] > 0
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

    # Normalise the data
    flat_v = v.reshape(v.shape[0], -1)
    mean = np.mean(flat_v, axis=0)
    std = np.std(flat_v, axis=0)
    flat_v = (flat_v - mean) / std

    flat_v = np.tanh(flat_v * 0.5)

    plot_multi_hists(
        data_list=flat_v,
        data_labels=k,
        col_labels=list(range(flat_v.shape[-1])),
        path=f"plots/{k}_distributions.png",
        bins=51,
        # logy=True,
    )
