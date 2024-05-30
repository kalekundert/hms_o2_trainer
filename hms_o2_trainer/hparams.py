import torch
import os

from .logging import log
from dataclasses import asdict
from itertools import product
from functools import partial
from collections.abc import Mapping

def make_hparams(factory, **kwargs):
    hparams = []
    keys = kwargs.keys()

    for values in product(*kwargs.values()):
        factory_kwargs = dict(zip(keys, values))
        hparams.append(factory(**factory_kwargs))

    return hparams

def label_hparams(key, *hparams):
    if isinstance(key, str):
        key = partial(interpolate, key)

    assert callable(key)

    return {
            key(x): x
            for x in hparams
    }

def require_hparams(key, hparams):
    if key is None:
        try:
            i = int(os.environ['SLURM_ARRAY_TASK_ID'])
            key = list(hparams)[i]

        except KeyError:
            digits = len(str(len(hparams) - 1))
            for i, known_key in enumerate(hparams):
                print(f'{i:>{digits}} {known_key}')
            raise SystemExit

    if key not in hparams:
        try:
            i = int(key)
            key = list(hparams)[i]
        except ValueError:
            pass

    log.info('using hyperparameters: %s', x := hparams[key])
    return key, x

def write_hparams(path, hparams, encoder=None):
    import json
    from os import makedirs

    if encoder:
        hparams = encoder(hparams)
    else:
        try:
            hparams = asdict(hparams)
        except TypeError:
            pass

    makedirs(path.parent, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(hparams, f)

def interpolate(template, obj):
    try:
        obj = asdict(obj)
    except TypeError:
        pass

    if isinstance(obj, Mapping):
        return template.format_map(obj)
    else:
        return template.format(obj)

def if_gpu(gpu_value, cpu_value):
    return gpu_value if torch.cuda.is_available() else cpu_value

