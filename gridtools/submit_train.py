import pyrootutils

root = pyrootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""

    standard_job_array(
        job_name="nuflows",
        work_dir=f"{root}/scripts",
        log_dir=f"{root}/logs",
        image_path="/home/users/l/leighm/scratch/Images/diffbeit-image_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=16,
        gpu_type="ampere",
        # vrap_per_gpu=20,
        time_hrs=24,
        mem_gb=40,
        opt_dict={
            "experiment": [
                "train_geant4.yaml",
            ],
            "trained_on": [
                "atlas_odd",
                "atlas_even",
            ],
        },
        use_dashes=False,
    )


if __name__ == "__main__":
    main()
