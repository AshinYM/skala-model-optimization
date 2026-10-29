"""One-time synthetic Compose initialization. Existing state is never retrained silently."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os
import tempfile


@contextmanager
def state_lock(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.bootstrap.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix='.ready-')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def ensure_demo(directory):
    from .api import load_model
    from .demo import make_demo
    from .tracking import verify_saved
    from .train import run

    state = Path(directory).resolve()
    with state_lock(state):
        marker = state/'demo-ready.json'
        output = state/'runs/demo-v1'
        model_path = output/'model.joblib'
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded.get('schema_version') != 1 or recorded.get('model_sha256') != hashlib.sha256(model_path.read_bytes()).hexdigest():
                raise ValueError('Existing state does not match its model; restore a backup instead of overwriting')
            load_model(model_path)
            verify_saved(output, state/'mlflow', 'compose-demo')
            return {'status': 'reused', 'data_origin': 'synthetic'}
        if not output.exists():
            data = make_demo(1000, seed=2026)
            payload = data.to_csv(index=False).encode()
            data_path = state/'data/demo.csv'
            data_path.parent.mkdir(parents=True, exist_ok=True)
            if data_path.exists() and data_path.read_bytes() != payload:
                raise ValueError('Existing demo input differs; refusing to overwrite it')
            if not data_path.exists():
                with data_path.open('xb') as handle:
                    handle.write(payload)
            run(data, output, dataset_sha256=hashlib.sha256(payload).hexdigest(),
                tracking_dir=state/'mlflow', experiment='compose-demo')
        # A crash after successful recording can be recovered without repeating training.
        required = [model_path, output/'report.json', output/'mlflow.json']
        if not all(path.is_file() for path in required):
            raise ValueError('Incomplete demo state; recover it or use a new volume. Nothing was overwritten')
        load_model(model_path)
        verify_saved(output, state/'mlflow', 'compose-demo')
        atomic_json(marker, {'schema_version': 1, 'data_origin': 'synthetic',
            'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'model_path': 'runs/demo-v1/model.joblib', 'experiment': 'compose-demo'})
        return {'status': 'initialized', 'data_origin': 'synthetic'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(ensure_demo(args.state_dir)))


if __name__ == '__main__':
    main()
