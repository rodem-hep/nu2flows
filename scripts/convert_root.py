import argparse
from glob import glob
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import rootutils
import uproot as ur
from numpy.lib.recfunctions import structured_to_unstructured as stu

root = rootutils.setup_root(search_from=__file__, pythonpath=True)


def awkward3D_to_padded(a: ak.Array, max_len: int | None = None) -> np.ndarray:
    """Convert an Awkward Array to a padded NumPy array.

    Parameters
    ----------
    a : ak.Array
        The input Awkward Array to be converted.
    max_len : int, optional
        The maximum length to pad/clipping.
        If None, it is determined by the maximum length of the arrays in `a`.

    Returns
    -------
    np.ndarray
        A NumPy array with the elements of `a` padded to `max_len`.
        Missing values filled with 0.0.

    """
    if max_len is None:
        max_len = int(ak.max(ak.num(a)))
    np_arr = ak.pad_none(a, target=max_len, clip=True)
    np_arr = ak.to_numpy(np_arr)
    np_arr = np.array(np_arr)
    return np.nan_to_num(np_arr, 0.0)


def init_dataset(
    file: h5py.File, group: str, name: str, data: np.ndarray, chunksize: int = 4000
) -> None:
    """Initialize a dataset within an HDF5 file.

    Parameters
    ----------
    file : h5py.File
        The HDF5 file object.
    group : str
        The name of the group within the HDF5 file where the dataset will be created.
    name : str
        The name of the dataset to be created.
    data : np.ndarray
        A sample of the data that will be stored in the dataset.
        Used to determine shape and dtype.
    chunksize : int, optional
        The size of chunks for the dataset. Defaults to 4000.
    """
    if name not in file[group]:
        file[group].create_dataset(
            name,
            shape=(0,) + data.shape[1:],
            maxshape=(None,) + data.shape[1:],
            chunks=(chunksize,) + data.shape[1:],
            dtype=data.dtype,
        )


def extend_dataset(file: h5py.File, group: str, name: str, data: np.ndarray) -> None:
    """Extend a dataset within an HDF5 file.

    Parameters
    ----------
    file : h5py.File
        The HDF5 file object.
    group : str
        The name of the group within the HDF5 file where the dataset will be extended.
    name : str
        The name of the dataset to be extended.
    data : np.ndarray
        The data to be appended to the dataset.
    """
    n = len(data)
    file[group][name].resize((file[group][name].shape[0] + n), axis=0)
    file[group][name][-n:] = data


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--input_dir",
        type=str,
        help="the path to input directory containing the root files",
    )
    args.add_argument(
        "--output_dir",
        type=str,
        help="directory for the output HDF5 file",
    )
    args.add_argument(
        "--output_name",
        type=str,
        help="name of the merged filr",
        default="combined.h5",
    )
    args.add_argument("--tree", type=str, help="input tree name", default="reco")
    args.add_argument("--chunksize", type=int, help="chunksize for the output file", default=4000)
    return args.parse_args()


def main():
    args = parse_args()

    print("Loading checkpoint")
    checkName = Path(args.output_dir, args.output_name).with_suffix(".txt")
    try:
        checkFiles = np.loadtxt(checkName, dtype=str)
        checkFiles = checkFiles.tolist()
        readMode = "a"
    except FileNotFoundError:
        checkFiles = []
        readMode = "w"

    print("Initialising the output HDF file")
    outName = Path(args.output_dir, args.output_name)
    outFile = h5py.File(outName, readMode)
    outFile.create_group("even")
    outFile.create_group("odd")

    print("Getting the list of input files")
    files = sorted(glob(str(Path(args.input_dir) / "*.root")))  # noqa
    files = [Path(f) for f in files]
    print(f"Found {len(files)} files")

    for i, file in enumerate(files):
        print(f" -> Processing {file.name} ({i + 1}/{len(files)})")

        # Skip files that have already been processed
        if str(file) in checkFiles:
            print(" -- already processed, skipping")
            continue

        # Whole thing in a try block to catch any errors
        try:
            inFile = ur.open(file)
            tree = inFile[args.tree]
            print(f" -- number of events: {tree.num_entries}")

            print(" -- getting the event info")
            evt_vars = [
                "eventNumber",
                "weight_pileup_NOSYS",
                "weight_beamspot",
                "weight_jvt_effSF_NOSYS",
                "weight_mc_NOSYS",
                "weight_btagSF_DL1dv01_Continuous_NOSYS",
                "weight_leptonSF_tight_NOSYS",
            ]
            evt_info = ak.to_numpy(tree.arrays(evt_vars))

            print(" -- loading MC info")
            mc_vars = [k for k in tree.keys() if "Ttbar_MC" in k]  # noqa
            mc_vars = [k for k in mc_vars if "FSR" not in k]
            mc_vars = [
                k for k in mc_vars if "_W_" not in k
            ]  # Dont need W's as we have their products
            mc_info = ak.to_numpy(tree.arrays(mc_vars))

            # Bjet weights being negative is a bug, abs them
            # This is a workaround until the source data is corrected
            evt_info["weight_btagSF_DL1dv01_Continuous_NOSYS"] = np.abs(
                evt_info["weight_btagSF_DL1dv01_Continuous_NOSYS"]
            )

            print(" -- converting misc")
            misc = tree["nuflows_input_misc_NOSYS"].array()
            misc = awkward3D_to_padded(misc).astype(np.float32)

            print(" -- converting MET")
            met = tree["nuflows_input_met_NOSYS"].array()
            met = awkward3D_to_padded(met).astype(np.float32)

            print(" -- converting leptons")
            lep = tree["nuflows_input_lep_NOSYS"].array()
            lep = awkward3D_to_padded(lep, 2).astype(np.float32)

            print(" -- converting jets")
            jet = tree["nuflows_input_jet_NOSYS"].array()
            jet = awkward3D_to_padded(jet, 10).astype(np.float32)
            jet[..., 3] = np.clip(jet[..., 3], 0, None)  # Jet mass and energy are fucked
            jet[..., 4] = np.clip(jet[..., 4], 0, None)

            print(" -- converting neutrinos")
            nu_1_vars = [
                "Ttbar_MC_Wdecay1_from_t_pt",
                "Ttbar_MC_Wdecay1_from_t_eta",
                "Ttbar_MC_Wdecay1_from_t_phi",
            ]
            nu_2_vars = [
                "Ttbar_MC_Wdecay2_from_tbar_pt",
                "Ttbar_MC_Wdecay2_from_tbar_eta",
                "Ttbar_MC_Wdecay2_from_tbar_phi",
            ]

            # This information should already be in the MC info
            nu_1 = stu(mc_info[nu_1_vars]).astype(np.float32)
            nu_2 = stu(mc_info[nu_2_vars]).astype(np.float32)
            neutrinos = np.dstack([nu_1, nu_2]).transpose((0, 2, 1))
            inFile.close()

            # Change the neutrino data to be px, py, pz and switch to GeV
            nu_pxpypz = np.zeros_like(neutrinos)
            nu_pxpypz[..., 0] = neutrinos[..., 0] * np.cos(neutrinos[..., 2])
            nu_pxpypz[..., 1] = neutrinos[..., 0] * np.sin(neutrinos[..., 2])
            nu_pxpypz[..., 2] = neutrinos[..., 0] * np.sinh(neutrinos[..., 1])
            nu_pxpypz /= 1000

            # Some event cleaning has to be done
            mask = ~np.any(nu_pxpypz[..., -1] == 0, axis=-1)
            mask &= ~np.any(nu_pxpypz > 1e6, axis=(-1, -2))
            mask &= ~np.any(np.isnan(nu_pxpypz), axis=(-1, -2))
            mask &= ~np.any(np.isnan(misc), axis=(-1, -2))
            mask &= ~np.any(np.isnan(met), axis=(-1, -2))
            mask &= ~np.any(np.isnan(lep), axis=(-1, -2))
            mask &= ~np.any(np.isnan(jet), axis=(-1, -2))

            # Apply the mask to all data, this also asserts that arrays are the same length
            nu_pxpypz = nu_pxpypz[mask]
            misc = misc[mask]
            met = met[mask]
            lep = lep[mask]
            jet = jet[mask]
            evt_info = evt_info[mask]
            mc_info = mc_info[mask]

            if i == 0:
                print(" -- initialising datasets")
                for split in ["even", "odd"]:
                    init_dataset(outFile, split, "neutrinos", nu_pxpypz)
                    init_dataset(outFile, split, "misc", misc)
                    init_dataset(outFile, split, "met", met)
                    init_dataset(outFile, split, "lep", lep)
                    init_dataset(outFile, split, "jet", jet)
                    init_dataset(outFile, split, "evt_info", evt_info)
                    init_dataset(outFile, split, "mc_info", mc_info)

            print(" -- saving")
            for split in ["even", "odd"]:
                save = evt_info["eventNumber"] % 2 == (split == "odd")
                extend_dataset(outFile, split, "neutrinos", nu_pxpypz[save])
                extend_dataset(outFile, split, "misc", misc[save])
                extend_dataset(outFile, split, "met", met[save])
                extend_dataset(outFile, split, "lep", lep[save])
                extend_dataset(outFile, split, "jet", jet[save])
                extend_dataset(outFile, split, "evt_info", evt_info[save])
                extend_dataset(outFile, split, "mc_info", mc_info[save])

            # Save the file name to the checkpoint
            checkFiles.append(str(file))
            np.savetxt(checkName, checkFiles, fmt="%s", delimiter="\n")

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    outFile.close()


if __name__ == "__main__":
    main()
