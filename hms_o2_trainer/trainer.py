import torch
import lightning as L
import inspect
import logging
import os

from .utils import is_slurm, log
from .requeue import RequeueBeforeTimeLimit
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

def get_trainer(
        hparams_name,
        *,
        dry_run=False,
        log_level=logging.INFO,
        float32_precision='high',
        out_dir=None,
        **trainer_kwargs,
):
    logging.basicConfig(level=log_level)

    # Lightning recommends setting this to either 'medium' or 'high' (as
    # opposed to 'highest', which is the default) when training on GPUs with
    # support for the necessary acceleration.  I don't think there's a good way
    # of knowing a priori what the best setting should be; so I chose the
    # 'high' setting as a compromise to be optimized later.
    torch.set_float32_matmul_precision(float32_precision)

    if is_slurm():
        hpc_callbacks = [RequeueBeforeTimeLimit()]
    else:
        hpc_callbacks = []

    if not out_dir:
        script_file = Path(inspect.currentframe().f_back.f_globals['__file__'])
        script_dir = script_file.parent

        out_dir = os.getenv('HOT_OUT_DIR', 'workspace').format(script_dir)
        out_dir = Path(out_dir).expanduser()

        if not out_dir.is_absolute():
            out_dir = script_dir / out_dir

    log.info("setting up trainer; float32_matmul_precision=%s output_dir=%s", float32_precision, out_dir)

    return L.Trainer(
            callbacks=[
                *hpc_callbacks,
                ModelCheckpoint(
                    save_last=True,
                    every_n_epochs=1,
                ),
            ],
            logger=TensorBoardLogger(
                save_dir=out_dir.parent,
                name=out_dir.name,
                version=hparams_name,
                default_hp_metric=False,
            ),
            fast_dev_run=(dry_run and 10),
            **trainer_kwargs,
    )
