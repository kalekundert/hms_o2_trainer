import os, sys
import logging

from pathlib import Path
from subprocess import run

log = logging.getLogger('hms_o2_trainer')
info = log.info

# Configuring logging is generally supposed to be left to applications, and not 
# done in libraries.  However, this library is meant to automate the process of 
# creating light-weight applications, so I think it's justified here.
logging.basicConfig(level=logging.INFO)

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
        
