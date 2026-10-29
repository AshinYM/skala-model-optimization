"""Offline state snapshots. Stop services first; restore only into an empty volume."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

from .bootstrap import state_lock

MANIFEST = '_archive_manifest.json'
MAX_BYTES = 128 * 1024 * 1024
MAX_FILES = 10000


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def backup(directory, output):
    root = Path(directory).resolve()
    with state_lock(root):
        if not (root/'demo-ready.json').is_file():
            raise ValueError('No complete demo state to snapshot')
        files = []
        for path in sorted(root.rglob('*')):
            if path.is_symlink() or (not path.is_file() and not path.is_dir()):
                raise ValueError('State must contain regular files/directories only')
            if path.is_file() and path.name != '.bootstrap.lock':
                files.append(path)
        if len(files) > MAX_FILES or sum(p.stat().st_size for p in files) > MAX_BYTES:
            raise ValueError('State exceeds snapshot limits')
        hashes = {p.relative_to(root).as_posix(): digest(p) for p in files}
        if MANIFEST in hashes:
            raise ValueError('Reserved manifest filename in state')
        manifest = json.dumps({'format': 1, 'files': hashes}, sort_keys=True).encode()
        if len(manifest) + sum(p.stat().st_size for p in files) > MAX_BYTES:
            raise ValueError('State plus manifest exceeds snapshot limits')
        with tarfile.open(fileobj=output, mode='w:gz') as archive:
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(manifest), 0o600
            archive.addfile(info, io.BytesIO(manifest))
            for path in files:
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        return {'files': len(files), 'consistent_snapshot_requires_stopped_services': True}


def restore(directory, source):
    root = Path(directory).resolve()
    with state_lock(root):
        if any(p.name != '.bootstrap.lock' for p in root.iterdir()):
            raise ValueError('Restore target must be empty; existing state is never overwritten')
        with tarfile.open(fileobj=source, mode='r:gz') as archive:
            members, total = [], 0
            for member in archive:
                total += member.size
                members.append(member)
                if len(members) > MAX_FILES+1 or total > MAX_BYTES:
                    raise ValueError('Archive exceeds restore limits')
            names = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or '..' in path.parts
                        or path.as_posix() != member.name or member.name in {'', '.', '.bootstrap.lock'}
                        or member.name in names):
                    raise ValueError('Unsafe or duplicate archive entry')
                if member.name != MANIFEST and path.parts[0] in {MANIFEST, '.bootstrap.lock'}:
                    raise ValueError('Reserved archive path')
                names.add(member.name)
            if MANIFEST not in names:
                raise ValueError('Archive manifest is required')
            entry = archive.getmember(MANIFEST)
            if entry.size > 2 * 1024 * 1024:
                raise ValueError('Manifest is too large')
            with archive.extractfile(entry) as handle:
                manifest = json.load(handle)
            hashes = manifest.get('files', {})
            if manifest.get('format') != 1 or not isinstance(hashes, dict) or set(hashes) != names-{MANIFEST}:
                raise ValueError('Archive content does not match its manifest')
            if not {'demo-ready.json', 'runs/demo-v1/model.joblib'} <= names:
                raise ValueError('Incomplete demo archive')
            with tempfile.TemporaryDirectory(prefix='.restore-', dir=root) as temporary:
                staging = Path(temporary)
                for name, expected in hashes.items():
                    target = staging/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(name) as reader, target.open('xb') as writer:
                        shutil.copyfileobj(reader, writer)
                    target.chmod(0o600)
                    if digest(target) != expected:
                        raise ValueError('Snapshot checksum mismatch')
                # Publish readiness last; validation/extraction failures leave the target empty.
                for path in sorted(staging.iterdir()):
                    if path.name != 'demo-ready.json':
                        path.rename(root/path.name)
                (staging/'demo-ready.json').rename(root/'demo-ready.json')
            return {'restored_files': len(hashes), 'state_mount_path_must_match_original': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['backup', 'restore'])
    parser.add_argument('--state-dir', required=True, type=Path)
    parser.add_argument('--services-stopped', action='store_true', required=True)
    args = parser.parse_args()
    if args.operation == 'backup':
        result = backup(args.state_dir, sys.stdout.buffer)
    else:
        # Docker stdin is a non-seekable pipe; stage the bounded compressed stream.
        with tempfile.TemporaryFile() as source:
            size = 0
            while chunk := sys.stdin.buffer.read(1024*1024):
                size += len(chunk)
                if size > MAX_BYTES+2*1024*1024:
                    raise ValueError('Compressed archive exceeds restore limit')
                source.write(chunk)
            source.seek(0)
            result = restore(args.state_dir, source)
    print(json.dumps(result), file=sys.stderr)


if __name__ == '__main__':
    main()
