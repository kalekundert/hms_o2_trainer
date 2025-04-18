import os

from .logging import log
from dataclasses import asdict
from itertools import product
from collections.abc import Mapping
from pathlib import Path

def make_hparams(factory, **kwargs):
    hparams = []
    keys = kwargs.keys()

    for values in product(*kwargs.values()):
        factory_kwargs = dict(zip(keys, values))
        hparams.append(factory(**factory_kwargs))

    return hparams

def require_hparams_index_from_cli(hparams, repr=repr):
    import docopt
    from __main__ import __file__, __doc__

    usage = f"""\
Usage:
    ./{Path(__file__).name} [<hparams>]
    ./{Path(__file__).name} (-h|--help)

Arguments:
    <hparams>
        The hyperparameters to use for this training run, specified as an  
        index number.  If the `$SLURM_ARRAY_TASK_ID` environment variable is 
        set (as it would be for an array job), it will be the default value for 
        this argument.  If no value is specified and no default is available, a 
        list of possible hyperparameters will be printed to the terminal.
"""

    if __doc__:
        usage = __doc__.strip() + '\n\n' + usage

    def maybe_int(x):
        if x is None:
            return None

        try:
            return int(x)
        except ValueError:
            raise ValueError(f"expected a hyperparameter index between 0-{len(hparams)-1}, not {i!r}")

    args = docopt.docopt(usage.strip())
    i = maybe_int(args['<hparams>'])
    return require_hparams(hparams, i, repr=repr), i

def require_hparams_from_cli(hparams, repr=repr):
    x, _ = require_hparams_index_from_cli(hparams, repr=repr)
    return x

def require_hparams(hparams, i, repr=repr):
    if i is None:
        try:
            i = int(os.environ['SLURM_ARRAY_TASK_ID'])

        except KeyError:
            digits = len(str(len(hparams) - 1))
            for i, hp in enumerate(hparams):
                print(f'{i:>{digits}} {repr(hp)}')
            raise SystemExit

    x = hparams[i]

    log.info('using hyperparameters: %s', repr(x))

    return x

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


