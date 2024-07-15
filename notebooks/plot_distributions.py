import h5py
import matplotlib.pyplot as plt
import numpy as np

file_dir = "/srv/beegfs/scratch/groups/dpnc/atlas/ttbar_vflows/data/rel24_240209/hdf5/"
file_name = "user.mleigh.410472.PhPy8EG.DAOD_PHYS.e6348_s3681_r13167_p5855.241007-v0_nuflowsout.h5"
file_path = file_dir + file_name

with h5py.File(file_path, "r") as f:
    jet = np.vstack([f["even"]["jet"][:], f["odd"]["jet"][:]])
    lep = np.vstack([f["even"]["lep"][:], f["odd"]["lep"][:]])
    met = np.vstack([f["even"]["met"][:], f["odd"]["met"][:]])
    nu = np.vstack([f["even"]["neutrinos"][:], f["odd"]["neutrinos"][:]])
    evt_info = np.concatenate([f["even"]["evt_info"][:], f["odd"]["evt_info"][:]], axis=0)

# Jets need to be padded so create a mask
jet_mask = ~np.all(jet == 0, axis=-1)

# Get the weight of the event
ws = [
    "weight_mc_NOSYS",
    "weight_pileup_NOSYS",
    "weight_beamspot",
    "weight_btagSF_DL1dv01_Continuous_NOSYS",
]

# Combine the weights by reducing the product
weights = np.cumprod([evt_info[w] for w in ws], axis=0).astype(np.float32)
weights = np.pad(weights, ((1, 0), (0, 0)), mode="constant", constant_values=1)


def plot_ax(weights, value, bins, ax, label):
    for i in range(len(weights)):
        w = np.repeat(weights[i, None].T, value.shape[-1], axis=-1)
        hist, bins = np.histogram(value, bins=bins, weights=w, density=True)
        ax.stairs(hist, bins, label=i)
        ax.set_xlabel(label)
        ax.set_yscale("log")


# Plot the raw distributions of the neutrinos
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Neutrino Energy
bins = np.linspace(0, 300, 50)
nu_e = np.linalg.norm(nu, axis=-1) / 1000
plot_ax(weights, nu_e, bins, axes[0, 0], "Neutrino Energy [GeV]")

# Jet Energy
bins = np.linspace(20, 1000, 50)
jet_e = np.exp(jet[..., 3]) / 1000
plot_ax(weights, jet_e, bins, axes[0, 1], "Jet Energy [GeV]")

# Electron Energy
bins = np.linspace(2, 500, 50)
lep_e = np.exp(lep[..., 3]) / 1000
plot_ax(weights, lep_e, bins, axes[1, 0], "Electron Energy [GeV]")

# SumET
bins = np.linspace(2, 1500, 50)
sumet = met[..., 2] / 1000
plot_ax(weights, sumet, bins, axes[1, 1], "SumET [GeV]")
axes[1, 1].legend()

fig.savefig("nu_energy.png")
plt.close()
