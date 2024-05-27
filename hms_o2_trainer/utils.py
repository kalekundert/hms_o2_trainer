import os
import re
import logging

log = logging.getLogger('hms_o2_trainer')
info = log.info

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
    m = re.fullmatch(r'BATCH_FLAG\s*\n(\d)\s*\n', p.stdout.decode())
    
    return m and m.group(1) == '1'

def get_job_id():
    array_job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    if array_job_id is not None:
        array_task_id = os.environ['SLURM_ARRAY_TASK_ID']
        return f'{array_job_id}_{array_task_id}'
    else:
        return os.environ['SLURM_JOB_ID']

def require_env(name):
    if name not in os.environ:
        raise ConfigError(f"must define ${name} environment variable")

class ConfigError(Exception):
    pass
