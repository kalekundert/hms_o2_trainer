import nestedtext as nt
from hms_o2_trainer import make_hparams, label_hparams, write_hparams
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

def test_label_hparams_str_int():
    assert label_hparams('i_{0}', 1, 2) == {'i_1': 1, 'i_2': 2}

def test_label_hparams_str_dataclass():
    @dataclass
    class HParams:
        x: int
        y: int

    assert label_hparams(
            'x_{x}_y_{y}',
            HParams(1, 2),
            HParams(3, 4),
    ) == {
            'x_1_y_2': HParams(1, 2),
            'x_3_y_4': HParams(3, 4),
    }

def test_label_hparams_callable():
    k = lambda i: f'i_{i + 1}'
    assert label_hparams(k, 1, 2) == {'i_2': 1, 'i_3': 2}

def test_write_hparams_dict(tmp_path):
    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, {'x': 1, 'y': 2})
    assert nt.load(p) == {'x': '1', 'y': '2'}

def test_write_hparams_dataclass(tmp_path):

    @dataclass
    class HParams:
        x: int
        y: int

    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, HParams(1, 2))
    assert nt.load(p) == {'x': '1', 'y': '2'}

def test_write_hparams_factory(tmp_path):
    p = tmp_path / 'hparams' / 'job_id.nt'
    write_hparams(p, (1, 2), lambda x: dict(zip("xy", x)))
    assert nt.load(p) == {'x': '1', 'y': '2'}



