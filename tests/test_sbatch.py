import hms_o2_trainer.sbatch as _hot
import parametrize_from_file as pff
import subprocess

from pathlib import Path

def run_sinfo(*args, **kwargs):
    sinfo_path = Path(__file__).parent / 'sinfo.stdout'
    return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=sinfo_path.read_text(),
    )


@pff.parametrize(
        schema=pff.cast(expected=eval),
)
def test_has_argv(argv, prefixes, expected):
    assert _hot.has_argv(argv, *prefixes) == expected

@pff.parametrize(
        schema=pff.error_or('expected'),
)
def test_pop_argv(argv, prefixes, expected, error):
    with error:
        i, key, value = _hot.pop_argv(argv, *prefixes)

        assert i == int(expected['i'])
        assert key == expected['key']
        assert value == expected['value']
        assert argv == expected['argv']

def test_parse_gpu_archs(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', run_sinfo)

    assert _hot.parse_gpu_archs() == {
            'compute-g-16-176':  'maxwell',
            'compute-g-16-177':  'kepler',
            'compute-g-16-254':  'volta',
            'compute-g-16-175':  'maxwell',
            'compute-g-16-194':  'kepler',
            'compute-g-16-255':  'volta',
            'compute-g-17-153':  'turing',
            'compute-g-17-145':  'turing',
            'compute-g-17-146':  'turing',
            'compute-g-17-147':  'volta',
            'compute-g-17-148':  'volta',
            'compute-g-17-149':  'volta',
            'compute-g-17-150':  'volta',
            'compute-g-17-151':  'volta',
            'compute-g-17-152':  'volta',
            'compute-g-17-154':  'turing',
            'compute-g-17-155':  'turing',
            'compute-g-17-156':  'turing',
            'compute-g-17-157':  'turing',
            'compute-g-17-158':  'turing',
            'compute-g-17-159':  'turing',
            'compute-g-17-160':  'turing',
            'compute-g-17-161':  'turing',
            'compute-g-17-162':  'ampere',
            'compute-g-17-163':  'ampere',
            'compute-g-17-164':  'ampere',
            'compute-g-17-165':  'ampere',
            'compute-g-17-166':  'lovelace',
            'compute-g-17-167':  'lovelace',
            'compute-g-17-168':  'lovelace',
            'compute-g-17-169':  'lovelace',
            'compute-g-17-170':  'lovelace',
            'compute-g-17-171':  'lovelace',
            'compute-gc-17-245': 'turing',
            'compute-gc-17-246': 'turing',
            'compute-gc-17-244': 'lovelace',
            'compute-gc-17-255': 'ampere',
            'compute-gc-17-247': 'turing',
            'compute-gc-17-248': 'maxwell',
            'compute-gc-17-249': 'ampere',
            'compute-gc-17-250': 'maxwell',
            'compute-gc-17-251': 'maxwell',
            'compute-gc-17-252': 'ampere',
            'compute-gc-17-253': 'ampere',
            'compute-gc-17-254': 'ampere',
            'compute-gc-17-241': 'lovelace',
            'compute-gc-17-242': 'lovelace',
            'compute-gc-17-243': 'lovelace',
            'compute-g-16-197':  'maxwell',
    }

def test_exclude_nodes_by_gpu_arch(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', run_sinfo)

    assert _hot.exclude_nodes_by_gpu_arch('ampere') == [
            'compute-g-16-176',
            'compute-g-16-177',
            'compute-g-16-254',
            'compute-g-16-175',
            'compute-g-16-194',
            'compute-g-16-255',
            'compute-g-17-153',
            'compute-g-17-145',
            'compute-g-17-146',
            'compute-g-17-147',
            'compute-g-17-148',
            'compute-g-17-149',
            'compute-g-17-150',
            'compute-g-17-151',
            'compute-g-17-152',
            'compute-g-17-154',
            'compute-g-17-155',
            'compute-g-17-156',
            'compute-g-17-157',
            'compute-g-17-158',
            'compute-g-17-159',
            'compute-g-17-160',
            'compute-g-17-161',
            'compute-gc-17-245',
            'compute-gc-17-246',
            'compute-gc-17-247',
            'compute-gc-17-248',
            'compute-gc-17-250',
            'compute-gc-17-251',
            'compute-g-16-197',
    ]

    assert _hot.exclude_nodes_by_gpu_arch('turing') == [
            'compute-g-16-176',
            'compute-g-16-177',
            'compute-g-16-254',
            'compute-g-16-175',
            'compute-g-16-194',
            'compute-g-16-255',
            'compute-g-17-147',
            'compute-g-17-148',
            'compute-g-17-149',
            'compute-g-17-150',
            'compute-g-17-151',
            'compute-g-17-152',
            'compute-gc-17-248',
            'compute-gc-17-250',
            'compute-gc-17-251',
            'compute-g-16-197',
    ]

    assert _hot.exclude_nodes_by_gpu_arch('volta') == [
            'compute-g-16-176',
            'compute-g-16-177',
            'compute-g-16-175',
            'compute-g-16-194',
            'compute-gc-17-248',
            'compute-gc-17-250',
            'compute-gc-17-251',
            'compute-g-16-197',
    ]

    assert _hot.exclude_nodes_by_gpu_arch('maxwell') == [
            'compute-g-16-177',
            'compute-g-16-194',
    ]

    assert _hot.exclude_nodes_by_gpu_arch('kepler') == []

