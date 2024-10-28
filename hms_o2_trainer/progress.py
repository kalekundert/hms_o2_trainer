import re
from pathlib import Path

def main():
    for path in Path.cwd().glob('*.err'):
        log = path.read_text()

        if m := re.match(r'.*_(\d+).err', path.name):
            job_id = m.group(1)
            print(f"Job ID:          {job_id}")

        if m := re.search(r'INFO:hms_o2_trainer:using hyperparameters: (.*)', log):
            hparams = m.group(1)
            print(f"Hyperparameters: {hparams}")
            
        epochs = [
                int(m.group(1))
                for line in log.splitlines()
                if (m := re.match(r'INFO: Epoch (\d+)', line))
        ]
        curr_epoch = max(epochs) + 1 if epochs else 0
        print(f"Epoch:           {curr_epoch}")
        print()

