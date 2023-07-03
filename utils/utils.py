"""General mix of utility functions not related to numpy or pytorch."""

import argparse
import json
import math
import operator
from functools import reduce
from pathlib import Path
from typing import Any, Mapping, Union

import yaml
from dotmap import DotMap
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


def standard_job_array(
    job_name: str,
    work_dir: str,
    image_path: str,
    command: str,
    log_dir: str,
    n_gpus: int,
    n_cpus: int,
    time_hrs: int,
    mem_gb: int,
    opt_dict: Mapping,
):
    # Calculate the total number of jobs to perform
    n_jobs = 1
    for key, vals in opt_dict.items():
        if not isinstance(vals, list):
            vals = [vals]
            opt_dict[key] = vals
        n_jobs *= len(vals)
    print(f"Generating gridsearch with {n_jobs} subjobs")

    # Creating the slurm submision file
    f = open(f"{job_name}.sh", "w", newline="\n", encoding="utf-8")
    f.write("#!/bin/sh\n\n")
    f.write(f"#SBATCH --cpus-per-task={n_cpus}\n")
    f.write(f"#SBATCH --mem={mem_gb}GB\n")
    f.write(f"#SBATCH --time={time_hrs}:00:00\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write(f"#SBATCH --output={log_dir}/%A_%a.out\n")
    if n_gpus:
        f.write(f"#SBATCH --gpus={n_gpus}\n")
        f.write("#SBATCH --partition=shared-gpu,private-dpnc-gpu\n")
    else:
        f.write("#SBATCH --partition=shared-cpu,private-dpnc-cpu\n")

    # The job array setup using the number of jobs
    f.write(f"\n#SBATCH -a 0-{n_jobs-1}\n\n")

    # Creating the bash lists of the job arguments
    for opt, vals in opt_dict.items():
        f.write(f"{opt}=(")
        for v in vals:
            f.write(" " + str(v))
        f.write(" )\n")
    f.write("\n")

    # The command line arguments
    f.write('export XDG_RUNTIME_DIR=""\n')
    f.write("module load GCC/9.3.0 Singularity/3.7.3-GCC-9.3.0-Go-1.14\n")

    # Creating the base singularity execution script
    f.write(f"cd {work_dir}\n")
    f.write("srun singularity exec --nv -B /srv,/home \\\n")
    f.write(f"   {image_path} \\\n")
    f.write(f"   {command} \\\n")

    # Now include the job array options using the bash lists
    run_tot = 1
    for opt, vals in opt_dict.items():
        f.write(f"       --{opt} ${{{opt}")
        f.write(f"[`expr ${{SLURM_ARRAY_TASK_ID}} / {run_tot} % {len(vals)}`]")
        f.write("} \\\n")
        run_tot *= len(vals)
    f.close()


def str2bool(mystring: str) -> bool:
    """Convert a string object into a boolean."""
    if isinstance(mystring, bool):
        return mystring
    if mystring.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if mystring.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def merge_dict(source: dict, update: dict) -> dict:
    """Merges two deep dictionaries recursively.

    - Apply to small dictionaries please!
    args:
        source: The source dict, will be copied (not modified)
        update: Will be used to overwrite and append values to the source
    """
    # Make a copy of the source dictionary
    merged = source.copy()

    # Cycle through all of the keys in the update
    for key in update:
        # If the key not in the source then add move on
        if key not in merged:
            merged[key] = update[key]
            continue

        # Check type of variable
        dict_in_upt = isinstance(update[key], dict)
        dict_in_src = isinstance(source[key], dict)

        # If neither are a dict, then simply replace the leaf variable
        if not dict_in_upt and not dict_in_src:
            merged[key] = update[key]

        # If both are dicts, then implement recursion
        elif dict_in_upt and dict_in_src:
            merged[key] = merge_dict(source[key], update[key])

        # Otherwise one is a dict and the other is a leaf, so fail!
        else:
            raise ValueError(
                f"Trying to merge dicts but {key} is a leaf node in one not other"
            )

    return merged


def print_dict(dic: dict, indent: int = 1) -> None:
    """Recursively print a dictionary using json.

    args:
        dic: The dictionary
        indent: The spacing/indent to do for nested dicts
    """
    print(json.dumps(dic, indent=indent))


def get_from_dict(data_dict: dict, key_list: list, default=None) -> Any:
    """Returns a value from a nested dictionary using list of keys."""
    try:
        return reduce(operator.getitem, key_list, data_dict)
    except KeyError:
        return default


def set_in_dict(data_dict: dict, key_list: list, value: Any):
    """Sets a value in a nested dictionary using a list of keys."""
    get_from_dict(data_dict, key_list[:-1])[key_list[-1]] = value


def key_prefix(pref: str, dic: dict) -> dict:
    """Adds a prefix to each key in a dictionary."""
    return {f"{pref}{key}": val for key, val in dic.items()}


def key_change(dic: dict, old_key: str, new_key: str, new_value=None) -> None:
    """Changes the key used in a dictionary inplace only if it exists."""

    # If the original key is not present, nothing changes
    if old_key not in dic:
        return

    # Use the old value and pop. Essentially a rename
    if new_value is None:
        dic[new_key] = dic.pop(old_key)

    # Both a key change AND value change. Essentially a replacement
    else:
        dic[new_key] = new_value
        del dic[old_key]


def remove_keys_starting_with(dic: dict, match: str) -> dict:
    """Removes all keys from the dictionary if they start with.

    - Returns a copy of the dictionary
    """
    return {key: val for key, val in dic.items() if key[: len(match)] != match}


def signed_angle_diff(angle1: Any, angle2: Any) -> Any:
    """Calculate diff between two angles reduced to the interval of [-pi,
    pi]"""
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi


def load_yaml_files(files: Union[list, tuple, str]) -> tuple:
    """Loads a list of files using yaml and returns a tuple of dictionaries."""

    # If the input is not a list then it returns a dict
    if isinstance(files, (str, Path)):
        with open(files, encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.Loader)

    opened = []

    # Load each file using yaml
    for fnm in files:
        with open(fnm, encoding="utf-8") as f:
            opened.append(yaml.load(f, Loader=yaml.Loader))

    return tuple(opened)


def save_yaml_files(
    path: str, file_names: Union[str, list, tuple], dicts: Union[dict, list, tuple]
) -> None:
    """Saves a collection of yaml files in a folder.

    - Makes the folder if it does not exist
    """

    # If the input is not a list then one file is saved
    if isinstance(file_names, (str, Path)):
        with open(f"{path}/{file_names}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(
                dicts.toDict() if isinstance(dicts, DotMap) else dicts,
                f,
                sort_keys=False,
            )
        return

    # Make the folder
    Path(path).mkdir(parents=True, exist_ok=True)

    # Save each file using yaml
    for f_nm, dic in zip(file_names, dicts):
        with open(f"{path}/{f_nm}.yaml", "w", encoding="UTF-8") as f:
            yaml.dump(
                dic.toDict() if isinstance(dic, DotMap) else dic, f, sort_keys=False
            )


def get_scaler(name: str):
    """Return a sklearn scaler object given a name."""
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "power":
        return PowerTransformer()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal")
    if name == "none":
        return None
    raise ValueError(f"No sklearn scaler with name: {name}")


def args_into_conf(
    argp: object, conf: dict, inpt_name: str, dest_keychains: Union[list, str] = None
) -> None:
    """Takes an input string and collects the attribute with that name from an
    object, then it places that value within a dictionary at certain locations
    defined by a list of destination keys chained together.

    This function is specifically designed for placing commandline arguments collected
    via argparse into to certain locations within a configuration dictionary

    There are some notable behaviours:
    - The dictionary is updated INPLACE!
    - If the input is not found on the obj or it is None, then the dict is not updated
    - If the keychain is a list the value is placed in multiple locations in the dict
    - If the keychain is None, then the input is placed in the first layer of the conf
      using its name as the key

    args:
        argp: The object from which to retrive the attribute using input_name
        conf: The dictionary to be updated with this new value
        input_name: The name of the value to retrive from the argument object
        dest_keychains: A string or list of strings for desinations in the dict
        (The keychain should show breaks in keys using '/')
    """

    # Exit if the input is not in the argp or if its value is None
    if not hasattr(argp, inpt_name) or getattr(argp, inpt_name) is None:
        return

    # Get the value from the argparse
    val = getattr(argp, inpt_name)

    # Do a simple replacement if the dest keychains is None
    if dest_keychains is None:
        conf[inpt_name] = val
        return

    # For a complex keychain we use a list for consistancy
    if isinstance(dest_keychains, str):
        dest_keychains = [dest_keychains]

    # Cycle through all of the destinations and place in the dictionary
    for dest in dest_keychains:
        set_in_dict(conf, dest.split("/"), val)
