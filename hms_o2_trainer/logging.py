import os, sys
import logging

from .utils import log
from pathlib import Path
from subprocess import run
from functools import cache

@cache
def init_logging():
    logging.basicConfig(level=logging.INFO)
    log_dependencies()

def log_dependencies():
    third_party_deps = os.environ.get('HOT_THIRD_PARTY_DEPS', '').split(':')

    for name, mod in sorted(sys.modules.items()):
        if name.startswith('_') or '.' in name:
            continue

        version = getattr(mod, '__version__', 'N/A')

        def log_third_party():
            if name in third_party_deps:
                log.info(
                        "report dependency version: name=%s version=%s",
                        name, version,
                )

        try:
            path = Path(mod.__file__)
        except (AttributeError, TypeError):
            log_third_party()
            continue


        git = 'git', '-C', str(path.parent)
        git_revparse = *git, 'rev-parse', 'HEAD'
        p = run(git_revparse, capture_output=True)

        if p.returncode != 0:
            log_third_party()
            continue

        commit = p.stdout.strip()

        git_diff_index = *git, 'diff-index', '--quiet', 'HEAD', '--'
        p = run(git_diff_index)
        dirty = (p.returncode != 0)

        log.info(
                "report dependency version: name=%s version=%s commit=%s dirty=%s",
                name, version, commit, dirty,
        )
        
