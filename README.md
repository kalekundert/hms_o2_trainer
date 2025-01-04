HMS O2 Trainer
==============

[![Last release](https://img.shields.io/pypi/v/hms_o2_trainer.svg)](https://pypi.python.org/pypi/hms_o2_trainer)
[![Python version](https://img.shields.io/pypi/pyversions/hms_o2_trainer.svg)](https://pypi.python.org/pypi/hms_o2_trainer)
[![Documentation](https://img.shields.io/readthedocs/hms_o2_trainer.svg)](https://hms-o2-trainer.readthedocs.io/en/latest/)
[![Test status](https://img.shields.io/github/actions/workflow/status/kalekundert/hms_o2_trainer/test.yml?branch=master)](https://github.com/kalekundert/hms_o2_trainer/actions)
[![Test coverage](https://img.shields.io/codecov/c/github/kalekundert/hms_o2_trainer)](https://app.codecov.io/github/kalekundert/hms_o2_trainer)
[![Last commit](https://img.shields.io/github/last-commit/kalekundert/hms_o2_trainer?logo=github)](https://github.com/kalekundert/hms_o2_trainer)

*HMS O2 Trainer* is a library for training machine-learning models on the HMS 
O2 cluster.  It handles logging, requeueing, resource requests, and setting 
some parameters to "best-practice" values, all while being completely agnostic 
to the model and data being used for training.

This library supports a workflow where you write a separate script for each set 
of hyperparameters to test.  These scripts should be very short; I think of 
them as powerful config files.  Here's a representative example:

```python
import hms_o2_trainer as hot
import sys

#SBATCH --time=1-0:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Note that all details relating to the model and the data are delegated to 
# these hypothetical third-party libraries.
from my_models import get_model
from my_data import get_data

HPARAMS = [
    dict(model='mlp', data='mnist'),
    dict(model='cnn', data='mnist'),
    dict(model='mlp', data='cifar'),
    dict(model='cnn', data='cifar'),
]

if __name__ == '__main__':
    hparams = hot.require_hparams_from_cli(HPARAMS)

    model = get_model(hparams['model'])
    data = get_data(hparams['data'])

    trainer = hot.get_trainer()
    trainer.fit(model, data)
```

Submit the above script to the cluster:
```bash
$ hot_sbatch --array 0-3 -- compare_mlp_cnn.py
```
