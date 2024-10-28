import os
import inspect

from contextlib import contextmanager
from more_itertools import first

KNOWN_METRICS = {
        'val/loss': 'min',
        'gen/accuracy': 'max',
        'gen/frechet_dist': 'min',
}

def get_trainer(
        *,
        dry_run=False,
        float32_precision='high',
        out_dir=None,
        version=None,
        ckpt_metric='val/loss',
        ckpt_mode=None,
        ckpt_top_k=1,
        **trainer_kwargs,
):
    import torch
    import lightning as L

    from .utils import is_slurm, get_job_id
    from .logging import log, log_dependencies
    from .requeue import RequeueBeforeTimeLimit
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    from pathlib import Path

    # Lightning recommends setting this to either 'medium' or 'high' (as
    # opposed to 'highest', which is the default) when training on GPUs with
    # support for the necessary acceleration.  I don't think there's a good way
    # of knowing a priori what the best setting should be; so I chose the
    # 'high' setting as a compromise to be optimized later.
    torch.set_float32_matmul_precision(float32_precision)

    if not ckpt_mode:
        ckpt_mode = KNOWN_METRICS[ckpt_metric]
    if not ckpt_mode:
        raise ValueError("must either (i) specify `ckpt_mode` or (ii) specify `ckpt_metric` with known mode")

    callbacks = [
            *trainer_kwargs.pop('callbacks', []),

            # This callback, in particular the `save_last=True` argument, is 
            # required for requeueing to work.
            ModelCheckpoint(
                monitor=ckpt_metric,
                mode=ckpt_mode,
                save_top_k=ckpt_top_k,
                save_last=True,
                every_n_epochs=1,
                verbose=True,
            ),
    ]

    if is_slurm():
        callbacks += [RequeueBeforeTimeLimit()]

    if not out_dir:
        script_file = Path(inspect.currentframe().f_back.f_globals['__file__'])

        out_dir = os.getenv('HOT_OUT_DIR', 'workspace').format(script_file)
        out_dir = Path(out_dir).expanduser()

        if not out_dir.is_absolute():
            out_dir = script_file.parent / out_dir

    if version is None and is_slurm():
        version = get_job_id()

    log.info("configure trainer: float32_matmul_precision=%s output_dir=%s version=%s", float32_precision, out_dir, version)

    class HmsO2Trainer(L.Trainer):

        def fit(self, *args, **kwargs):
            log_dependencies()
            kwargs = dict(ckpt_path='last') | kwargs
            return super().fit(*args, **kwargs)

    return HmsO2Trainer(
            callbacks=callbacks,
            logger=TensorBoardLogger(
                save_dir=out_dir.parent,
                name=out_dir.name,
                version=version,
                default_hp_metric=False,
            ),
            fast_dev_run=(dry_run and 10),
            **trainer_kwargs,
    )

def show_layers(model, data, **kwargs):
    from torchtnt.utils.module_summary import (
            get_module_summary, get_summary_table
    )

    data.setup('fit')
    x = first(data.train_dataloader())

    # Run the model once, to make sure any uninitialized buffers are 
    # materialized.
    model(x)
    model.eval()

    summary = get_module_summary(model, x)

    # The `summary` object is pretty easy to work with.  I can format my own 
    # table, more nicely.
    table = get_summary_table(summary)
    print(table)

def show_dag(model, data, **kwargs):
    """
    Produce a graph of every operation that occurs during a forward pass of the 
    model.

    This graph is produced by the `torchlens` library.  This can be a pretty 
    expensive operation, so I recommend using the smallest batch size possible.

    This function automatically puts the model in eval mode.  This is important 
    for ESCNN models, because in training mode the convolutional layers do a 
    bunch of extra calculations that take a long time and clutter the final 
    graph.
    """
    import torchlens as tl

    model.eval()
    data.setup('fit')

    x = first(data.train_dataloader())
    tl.show_model_graph(model, [x], **kwargs)

@contextmanager
def show_memory(snapshot_path: str = 'cuda_mem.pkl'):
    import torch.cuda
    from humanize import naturalsize as bytes

    torch.cuda.memory._record_memory_history()

    try:
        yield

    finally:
        torch.cuda.memory._dump_snapshot(snapshot_path)

        print('Max VRAM Allocated:', bytes(torch.cuda.max_memory_allocated()))
        print('Max VRAM Reserved: ', bytes(torch.cuda.max_memory_reserved()))
