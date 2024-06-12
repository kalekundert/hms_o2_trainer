import torch
import os

from .logging import log
from dataclasses import asdict
from itertools import product
from functools import partial
from collections.abc import Mapping
from pathlib import Path

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

def require_hparams_from_cli(hparams):
    import docopt
    from __main__ import __file__, __doc__

    usage = f"""\
Usage:
    ./{Path(__file__).name} [<hparams>]
    ./{Path(__file__).name} (-h|--help)

Arguments:
    <hparams>
        The hyperparameters to use for this training run, specified either as a 
        name or an index number.  If the `$SLURM_ARRAY_TASK_ID` environment 
        variable is set (as it would be for an array job), it will be the 
        default value for this argument.  If no value is specified and no 
        default is available, a list of possible hyperparameters will be 
        printed to the terminal.
"""

    if __doc__:
        usage = __doc__.strip() + '\n\n' + usage

    args = docopt.docopt(usage.strip())
    return require_hparams(args['<hparams>'], hparams)

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

    # try:
    #     job_id = get_job_id()
    # except KeyError:
    #     pass
    # else:
    #     write_hparams(Path('hparams') / f'{job_id}.json', x)

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

    log.info('record hyperparameters: %s', path)

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

