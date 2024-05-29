import subprocess
import hms_o2_trainer as hot

def test_is_sbatch_no_job_id(monkeypatch):
    assert not hot.is_sbatch()

def test_is_sbatch_single_row(monkeypatch):

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=b'BATCH_FLAG          \n1                   \n',
        )

    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setattr(subprocess, 'run', run)

    assert hot.is_sbatch()

def test_is_sbatch_multiple_rows(monkeypatch):

    def run(*args, **kwargs):
        return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=b'BATCH_FLAG          \n1                   \n1                   \n',
        )

    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.setattr(subprocess, 'run', run)

    assert hot.is_sbatch()

