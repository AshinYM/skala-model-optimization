import io
import importlib.util
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from churn.bootstrap import ensure_demo, state_lock
from churn.compose_ops import require_stopped
from churn.state_archive import backup, restore


@unittest.skipUnless(os.name == 'posix', 'Container maintenance uses POSIX file locks')
class ComposeStateTests(unittest.TestCase):
    def fixture(self, root):
        (root/'runs/demo-v1').mkdir(parents=True)
        (root/'demo-ready.json').write_text('{"schema_version":1}')
        (root/'runs/demo-v1/model.joblib').write_bytes(b'trusted test fixture, not an executable model')
        (root/'mlflow').mkdir()
        (root/'mlflow/tracking.db').write_bytes(b'snapshot fixture')

    @unittest.skipUnless(importlib.util.find_spec('fastapi'), 'Install requirements-api.txt for demo bootstrap tests')
    def test_bootstrap_is_idempotent_and_detects_changed_model(self):
        def training(data, output, **kwargs):
            output.mkdir(parents=True)
            (output/'model.joblib').write_bytes(b'fixed model')
            (output/'report.json').write_text('{}')
            (output/'mlflow.json').write_text('{}')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch('churn.train.run', side_effect=training) as train, \
                 patch('churn.api.load_model'), patch('churn.tracking.verify_saved'):
                self.assertEqual(ensure_demo(root)['status'], 'initialized')
                marker = (root/'demo-ready.json').read_bytes()
                self.assertEqual(ensure_demo(root)['status'], 'reused')
                self.assertEqual(train.call_count, 1)
                self.assertEqual((root/'demo-ready.json').read_bytes(), marker)
                (root/'runs/demo-v1/model.joblib').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    ensure_demo(root)
                self.assertEqual(train.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec('fastapi'), 'Install requirements-api.txt for demo bootstrap tests')
    def test_incomplete_initialization_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'runs/demo-v1'
            output.mkdir(parents=True)
            (output/'model.joblib').write_bytes(b'keep')
            with patch('churn.train.run') as train:
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    ensure_demo(root)
                train.assert_not_called()
            self.assertEqual((output/'model.joblib').read_bytes(), b'keep')

    def test_state_lock_prevents_concurrent_initializer(self):
        with tempfile.TemporaryDirectory() as directory:
            with state_lock(directory):
                with self.assertRaises(BlockingIOError):
                    with state_lock(directory):
                        pass
            with state_lock(directory):
                pass

    def test_archive_roundtrip_preserves_all_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root/'original', root/'restored'
            self.fixture(source)
            stream = io.BytesIO()
            count = backup(source, stream)['files']
            stream.seek(0)
            self.assertEqual(restore(target, stream)['restored_files'], count)
            for file in source.rglob('*'):
                if file.is_file() and file.name != '.bootstrap.lock':
                    self.assertEqual(file.read_bytes(), (target/file.relative_to(source)).read_bytes())

    def test_restore_does_not_overwrite_existing_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root/'original')
            (root/'target').mkdir()
            (root/'target/keep').write_bytes(b'keep')
            stream = io.BytesIO()
            backup(root/'original', stream)
            stream.seek(0)
            with self.assertRaisesRegex(ValueError, 'must be empty'):
                restore(root/'target', stream)
            self.assertEqual((root/'target/keep').read_bytes(), b'keep')

    def test_unsafe_archive_paths_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, kind in [('../outside', tarfile.REGTYPE), ('/outside', tarfile.REGTYPE),
                               ('link', tarfile.SYMTYPE), ('link', tarfile.LNKTYPE),
                               ('.bootstrap.lock/overwrite', tarfile.REGTYPE)]:
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w:gz') as archive:
                    entry = tarfile.TarInfo(name)
                    entry.type, entry.linkname = kind, '/outside'
                    archive.addfile(entry, io.BytesIO(b''))
                stream.seek(0)
                target = Path(directory)/'target'
                with self.assertRaises(ValueError):
                    restore(target, stream)
                self.assertEqual({f.name for f in target.iterdir()}, {'.bootstrap.lock'})

    def test_corrupted_archive_is_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root/'original')
            good = io.BytesIO()
            backup(root/'original', good)
            good.seek(0)
            broken = io.BytesIO()
            with tarfile.open(fileobj=good, mode='r:gz') as source, tarfile.open(fileobj=broken, mode='w:gz') as target:
                for member in source.getmembers():
                    value = source.extractfile(member).read()
                    if member.name.endswith('model.joblib'):
                        value = b'changed'
                        member.size = len(value)
                    target.addfile(member, io.BytesIO(value))
            broken.seek(0)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore(root/'restored', broken)
            self.assertEqual({p.name for p in (root/'restored').iterdir()}, {'.bootstrap.lock'})

    def test_backup_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root/'link').symlink_to(root/'demo-ready.json')
            with self.assertRaisesRegex(ValueError, 'regular files'):
                backup(root, io.BytesIO())

    def test_running_services_block_snapshot_operations(self):
        for response in ['[{"State":"running"}]', '{"State":"running"}\n{"State":"running"}']:
            with patch('subprocess.check_output', return_value=response):
                with self.assertRaisesRegex(ValueError, 'Stop this Compose project'):
                    require_stopped(['docker', 'compose'])
        with patch('subprocess.check_output', return_value='[]'):
            require_stopped(['docker', 'compose'])


if __name__ == '__main__':
    unittest.main()
