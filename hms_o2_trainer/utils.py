import torch
import os

def is_slurm():
    return 'SLURM_JOB_ID' in os.environ

def is_sbatch():
    from subprocess import run

    try:
        job_id = os.environ['SLURM_JOB_ID']
    except KeyError:
        return False

    squeue = 'squeue', '--Format=batchflag', '--jobs', job_id
    p = run(squeue, capture_output=True)

    try:
        head, *rows = (x.strip() for x in p.stdout.decode().splitlines() if x)
    except:
        # One of my long-running jobs failed on this line, and I'm not sure 
        # why.  Hoping to catch some better debugging information next time 
        # this happens.
        print(f'$ {" ".join(squeue)}\n{p.stdout!r}')
        raise

    return head == 'BATCH_FLAG' and all(x == '1' for x in rows)

def if_gpu(gpu_value, cpu_value):
    return gpu_value if torch.cuda.is_available() else cpu_value

def get_vram_gb():
    vram_free, vram_tot = torch.cuda.mem_get_info()
    return vram_tot / 1e9

def get_job_id():
    # Note that for array jobs, `squeue` displays job ids in the format `<id of 
    # first job in array>_<task id>`.  These aren't really job ids, though, so 
    # we don't need to worry about recapitulating them here.
    return os.environ['SLURM_JOB_ID']

def require_env(name):
    if name not in os.environ:
        raise ConfigError(f"must define ${name} environment variable")

class ConfigError(Exception):
    pass
