import json
from hms_o2_trainer import make_hparams, write_hparams
from dataclasses import dataclass

def test_make_hparams():
    @dataclass
    class HParams:
        x: int
        y: int

    hparams = make_hparams(HParams, x=[1, 2], y=[3,4,5])

    assert hparams == [
            HParams(1, 3),
            HParams(1, 4),
            HParams(1, 5),
            HParams(2, 3),
            HParams(2, 4),
            HParams(2, 5),
    ]

def test_write_hparams_dict(tmp_path):
    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, {'x': 1, 'y': 2})

    with open(p) as f:
        assert json.load(f) == {'x': 1, 'y': 2}

def test_write_hparams_dataclass(tmp_path):

    @dataclass
    class HParams:
        x: int
        y: int

    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, HParams(1, 2))

    with open(p) as f:
        assert json.load(f) == {'x': 1, 'y': 2}

def test_write_hparams_factory(tmp_path):
    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, (1, 2), lambda x: dict(zip("xy", x)))

    with open(p) as f:
        assert json.load(f) == {'x': 1, 'y': 2}



