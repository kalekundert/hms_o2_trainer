"""\
Submit training jobs on a SLURM cluster.

- Requeueing is automatically configured.
- Parameters that don't typically change between runs (e.g. GPU/Partition/QOS) 
  are read from environment variables, so they don't need to be specified every 
  time.
- The proper virtual environment can be loaded before the job starts.

Usage:
    hot_sbatch [-d] [-h] [<sbatch args>]... -- <script> [<script args>]...

Arguments:
    <sbatch args>
        Arguments to pass directly on to sbatch

    <script>
        The path to the script to run on the cluster.  This script may contain 
        `sbatch` options preceded by `#SBATCH`; these options will be added to 
        the `sbatch` command.

    <script args>
        Any extra arguments to pass to the script.

Options:
    -h --help
        Print this usage information.

    -d --dry-run
        Print the sbatch command that would be used, but don't run it.

Environment Variables:
    The following environment variables must be defined, unless the 
    corresponding option is specified:

        HOT_SBATCH_GRES:
            The `--gres` option for `sbatch`.
        
        HOT_SBATCH_PARTITION:
            The `--partition` option for `sbatch`.

        HOT_SBATCH_QOS:
            The `--qos` option for `sbatch`.

        HOT_SETUP_ENV:
            A script that will be sourced before the training command is run.  
            This script should make sure that all the relevant modules/virtual 
            environments are setup.

    The following environment variables may be defined:

        HOT_SBATCH_OUTPUT:
            The `--output` option for `sbatch`.  If not specified, the default 
            is `%x_%j.out`.

        HOT_SBATCH_ERROR:
            The `--error` option for `sbatch`.  If not specified, the default 
            is `%x_%j.err`.
"""

import re
import os
import sys

from .utils import require_env
from subprocess import run
from pathlib import Path
from itertools import product

def main():
    args = parse_cli_args(__doc__, sys.argv[1:])
    argv = args['<sbatch args>']
    script = Path(args['<script>'])
    script_args = args['<script args>']

    argv += find_sbatch_comments(script)

    sbatch = ['sbatch', *argv]

    if not has_argv(argv, '--gres'):
        require_env('HOT_SBATCH_GRES')
        sbatch += ['--gres', os.environ['HOT_SBATCH_GRES']]
    if not has_argv(argv, '-p', '--partition'):
        require_env('HOT_SBATCH_PARTITION')
        sbatch += ['--partition', os.environ['HOT_SBATCH_PARTITION']]
    if not has_argv(argv, '-q', '--qos'):
        require_env('HOT_SBATCH_QOS')
        sbatch += ['--qos', os.environ['HOT_SBATCH_QOS']]
    if not has_argv(argv, '-J', '--job-name'):
        sbatch += ['--job-name', script.stem]
    if not has_argv(argv, '--signal'):
        sbatch += ['--signal', 'B:USR1']
    if not has_argv(argv, '--no-requeue'):
        sbatch += ['--requeue']
    if not has_argv(argv, '--open-mode'):
        sbatch += ['--open-mode', 'append']
    if not has_argv(argv, '-o', '--output'):
        sbatch += ['--output', os.environ.get('HOT_SBATCH_OUTPUT', '%x_%j.out')]
    if not has_argv(argv, '-e', '--error'):
        sbatch += ['--error', os.environ.get('HOT_SBATCH_ERROR', '%x_%j.err')]

    require_env('HOT_SETUP_ENV')
    sbatch += [
            Path(__file__).parent / 'train.sbatch',
            script,
            *script_args,
    ]

    if args['--dry-run']:
        sbatch = ['echo'] + sbatch

    run(map(str, sbatch))

def parse_cli_args(usage, argv):
    all_args = {}
    sbatch_args = []
    script_args = []
    curr_args = sbatch_args

    for arg in argv:
        if arg == '--':
            curr_args = script_args
        else:
            curr_args.append(arg)

    def pop_flag(*flags):
        has_flag = False

        for flag in flags:
            try:
                i = sbatch_args.index(flag)
            except ValueError:
                continue

            has_flag = True
            del sbatch_args[i]

        return has_flag

    if pop_flag('-h', '--help'):
        print(usage, file=sys.stderr)
        raise SystemExit

    if not script_args:
        print("must specify a script to submit to the cluster", file=sys.stderr)
        print("did you forget to put `--` before the script?", file=sys.stderr)
        raise SystemExit

    all_args['--dry-run'] = pop_flag('-d', '--dry-run')
    all_args['<sbatch args>'] = sbatch_args
    all_args['<script>'] = script_args[0]
    all_args['<script args>'] = script_args[1:]

    return all_args

def has_argv(argv, *prefixes):
    return any(x.startswith(p) for x, p in product(argv, prefixes))

def find_sbatch_comments(path):
    sbatch_comment = r'\s*#SBATCH (.*)'
    args = []

    for line in path.read_text().splitlines():
        if m := re.match(sbatch_comment, line):
            args.append(m.group(1))

    return args


if __name__ == '__main__':
    main()
