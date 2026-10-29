"""Host-side Compose backup/restore; refuses to run while this project's services run."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess


def compose_args(project=None):
    root = Path(__file__).resolve().parents[1]
    command = ['docker', 'compose', '--project-directory', str(root)]
    if project:
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', project):
            raise ValueError('Invalid Compose project name')
        command += ['--project-name', project]
    return command


def require_stopped(command):
    raw = subprocess.check_output(command+['ps', '--status', 'running', '--format', 'json'], text=True).strip()
    if raw:
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if rows:
            raise ValueError('Stop this Compose project before snapshot or restore')


def transfer(operation, archive, project=None):
    command = compose_args(project)
    require_stopped(command)
    runner = command+['run', '--rm', '-T', '--no-deps', '--pull', 'never', 'init',
                      'python', '-m', 'churn.state_archive', operation, '--state-dir', '/state', '--services-stopped']
    archive = Path(archive)
    if operation == 'backup':
        archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as handle:
                subprocess.run(runner, stdout=handle, check=True)
        except BaseException:
            archive.unlink(missing_ok=True)
            raise
    else:
        with archive.open('rb') as handle:
            subprocess.run(runner, stdin=handle, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['backup', 'restore'])
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--project-name')
    args = parser.parse_args()
    transfer(args.operation, args.archive, args.project_name)
    print(args.operation+' complete')


if __name__ == '__main__':
    main()
