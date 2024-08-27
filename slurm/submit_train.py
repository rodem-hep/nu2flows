import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array


def main() -> None:
    """Main executable script."""
    standard_job_array(
        job_name="nuflows",
        work_dir=f"{root}/scripts",
        log_dir=f"{root}/logs",
        image_path="/srv/fast/share/rodem/images/nu2flows_latest.sif",
        command="python train.py",
        n_gpus=1,
        n_cpus=16,
        gpu_type="ampere",
        vram_per_gpu=20,
        time_hrs=72,
        mem_gb=40,
        opt_dict={
            "experiment": "train_geant4.yaml",
            "train_group": [
                "even",
                "odd",
            ],
            "test_group": [
                "odd",
                "even",
            ],
        },
        is_grid=False,
        use_dashes=False,
    )


if __name__ == "__main__":
    main()
