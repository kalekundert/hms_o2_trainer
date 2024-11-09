"""\
Submit training jobs on a SLURM cluster.

Usage:
    hot_sbatch [-d] [-h] [<sbatch args>]... -- <script> [<script args>]...

Description:
    This command is a relatively thin wrapper around the normal `sbatch` 
    command, with the goal of providing some more useful defaults for the 
    specific task of training machine learning models.  For example:

    - Requeueing is automatically configured.
    - Parameters that don't typically change between runs (e.g. 
      GPU/Partition/QOS) are read from environment variables, so they don't 
      need to be specified every time.
    - The proper virtual environment can be loaded before the job starts.

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

    --gpu-arch <arch>
        Specify the minimum GPU architecture to be used for this job.  Only 
        nodes with the requested architecture or a more recent one will be 
        allowed to run the job.  Below are the architecture that can be 
        specified, listed from oldest (top left) to newest (bottom right):

            tesla       fermi       kepler      maxwell     pascal
            volta       turing      ampere      hopper      lovelace

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

        HOT_THIRD_PARTY_DEPS:
            A colon-separated list of python package names.  If any of these 
            packages are imported (directly or indirectly) by the training 
            script, their version numbers will be logged.  Note that any 
            packages that are present as git repositories on the current system 
            will automatically be logged, regardless of this setting.  (Such 
            packages are considered "first party".)
"""

import re
import os
import sys
import subprocess

from .utils import ConfigError, require_env
from pathlib import Path
from itertools import product

NOT_SPECIFIED = object()

def main():
    try:
        args = parse_cli_args(__doc__, sys.argv[1:])
        argv = args['<sbatch args>']
        script = Path(args['<script>']).resolve()
        script_args = args['<script args>']

        # Put the arguments after the comments, so that arguments supersede 
        # comments (just like they do for "real" `sbatch`).
        argv = find_sbatch_comments(script) + argv

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

        exclude = []

        if has_argv(argv, '--gpu-arch'):
            if has_argv(argv, '--exclude'):
                raise ConfigError("cannot specify `--gpu-arch` and `--exclude` together")
            _, _, arch = pop_argv(sbatch, '--gpu-arch')
            exclude += exclude_nodes_by_gpu_arch(arch)

        if not has_argv(argv, '--exclude'):
            exclude += get_blacklisted_nodes()
            sbatch += ['--exclude', ','.join(exclude)]

        require_env('HOT_SETUP_ENV')
        sbatch += [
                Path(__file__).parent / 'train.sbatch',
                script,
                *script_args,
        ]

        if args['--dry-run']:
            sbatch = ['echo'] + sbatch

        subprocess.run(map(str, sbatch))

    except ConfigError as err:
        print("error:", err, file=sys.stderr)
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(2)

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

def pop_argv(argv, *prefixes):
    key = None

    for i, arg in enumerate(argv):
        for prefix in prefixes:
            if not arg.startswith(prefix):
                continue

            # Don't stop once we find a match.  We might find a later match, 
            # and if so, we want to use that instead.
            if arg.startswith(prefix + '='):
                key, value = arg.split('=', maxsplit=1)
                begin, end = i, i + 1
            else:
                key, value = arg, argv[i + 1]
                begin, end = i, i + 2

    if key is None:
        raise KeyError(prefixes)

    argv[:] = argv[:begin] + argv[end:]
    return begin, key, value

def find_sbatch_comments(path):
    sbatch_comment = r'\s*#SBATCH (.*)'
    args = []

    for line in path.read_text().splitlines():
        if m := re.match(sbatch_comment, line):
            args.append(m.group(1))

    return args

def get_blacklisted_nodes():
    return [
            # 2024/09/19: These nodes give the following error:
            #   
            #   RuntimeError: CUDA unknown error - this may be due to an 
            #   incorrectly set up environment, e.g. changing env variable 
            #   CUDA_VISIBLE_DEVICES after program start. Setting the available 
            #   devices to be zero.
            #
            # An internet search indicates that restarting the nvidia driver 
            # can often fix this, but I don't have permission to do that on the 
            # cluster.
            'compute-gc-17-244',
            'compute-gc-17-249',
    ]

def exclude_nodes_by_gpu_arch(arch):
    arch_ranks = [
            'tesla',
            'fermi',
            'kepler',
            'maxwell',
            'pascal',
            'volta',
            'turing',
            'ampere',
            'hopper',
            'lovelace',
    ]
    i = arch_ranks.index(arch)
    exclude_archs = arch_ranks[:i]
    exclude_nodes = []

    for node, arch in parse_gpu_archs().items():
        if arch in exclude_archs:
            exclude_nodes.append(node)

    return exclude_nodes

def parse_gpu_archs():
    sinfo = 'sinfo', '--Format=nodehost,gres:80'
    p = subprocess.run(sinfo, capture_output=True, text=True)

    archs = {
            'a100': 'ampere',
            'a100.mig': 'ampere',
            'a40': 'ampere',
            'a6000': 'ampere',
            'l40': 'lovelace',
            'l40s': 'lovelace',
            'rtx6000': 'turing',
            'rtx8000': 'turing',
            'teslaK80': 'kepler',
            'teslaM40': 'maxwell',
            'teslaV100': 'volta',
            'teslaV100s': 'volta',
            'titanx': 'maxwell',
    }
    nodes = {}

    for line in p.stdout.splitlines():
        node, gres = line.split(maxsplit=1)
        if gres.strip() in ['GRES', '(null)']:
            continue

        card = gres.split(':')[1]
        nodes[node] = archs[card]

    return nodes

if __name__ == '__main__':
    main()
