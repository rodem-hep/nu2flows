import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from mltools.mltools.utils import standard_job_array

data_dir = "/srv/beegfs/scratch/groups/dpnc/atlas/ttbar_vflows/data/rel24_240209/"
f = "user.mleigh.410472.PhPy8EG.DAOD_PHYS.*.17-07-24-v0_nuflowsout/"
opt_dict = {
    "input_dir": data_dir + "root/" + f,
    "output_dir": data_dir + "hdf5/",
    "tree": "reco",
}

standard_job_array(
    job_name="convert",
    work_dir=root / "scripts",
    log_dir=root / "logs",
    image_path="/srv/fast/share/rodem/images/jetssl_latest.sif",
    command="python convert.py",
    opt_dict=opt_dict,
    n_cpus=4,
    time_hrs=1,
    mem_gb=8,
    is_grid=False,
)
