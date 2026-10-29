import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from churn.demo import make_demo
from churn.features import pipeline
from churn.tracking import configure, local_artifacts, log_training, probe, verify_saved
from churn.train import run


class LocalArtifactTests(unittest.TestCase):
    def test_remote_or_outside_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(local_artifacts((root/'models').as_uri(), root), (root/'models').resolve())
            for value in ['https://example.invalid/artifacts', 's3://bucket/models',
                          'file://other-host/models', root.parent.as_uri()]:
                with self.assertRaises(ValueError):
                    local_artifacts(value, root)


@unittest.skipUnless(importlib.util.find_spec('mlflow'), 'Install requirements-tracking.txt for local MLflow integration tests')
class TrackingTests(unittest.TestCase):
    def test_local_tracking_registry_and_no_customer_samples(self):
        data = make_demo(240)
        sentinel = 'DO_NOT_LOG_CUSTOMER_SENTINEL'
        data['Surname'] = sentinel
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, store = root/'run', root/'tracking'
            with patch.dict('os.environ', {'MLFLOW_TRACKING_URI': 'https://example.invalid',
                                          'MLFLOW_REGISTRY_URI': 'https://example.invalid'}):
                report = run(data, output, model_candidates={
                    'lr_base': pipeline(LogisticRegression(max_iter=1000), False),
                    'lr_fe': pipeline(LogisticRegression(max_iter=1000), True)}, tracking_dir=store)
            metadata = json.loads((output/'mlflow.json').read_text())
            self.assertEqual(metadata['artifact_max_abs_error'], 0)
            self.assertEqual(metadata['registry_max_abs_error'], 0)
            self.assertEqual(verify_saved(output, store)['max_abs_error'], 0)
            import mlflow.pyfunc
            expected = joblib.load(output/'model.joblib')['pipeline'].predict_proba(probe())
            actual = mlflow.pyfunc.load_model(metadata['model_uri']).predict(probe())
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            client, experiment_id = configure(store)
            runs = client.search_runs([experiment_id])
            self.assertEqual(len(runs), 3)
            children = [r for r in runs if r.data.tags.get('phase') == 'candidate']
            self.assertEqual(len(children), 2)
            self.assertTrue(all(not any(k.startswith('test_') for k in r.data.metrics) for r in children))
            parent = client.get_run(metadata['run_id'])
            self.assertEqual(parent.data.metrics['test_roc_auc'], report['final_test']['roc_auc'])
            self.assertTrue(all(r.data.tags['mlflow.parentRunId'] == parent.info.run_id for r in children))
            version = client.get_model_version('BankChurnPipeline', metadata['registered_version'])
            self.assertEqual(version.tags['roundtrip_verified'], 'true')
            self.assertEqual(version.tags['deployment'], 'not_deployed')
            self.assertFalse(client.get_registered_model('BankChurnPipeline').aliases)
            for file in (store/'artifacts').rglob('*'):
                if file.is_file():
                    self.assertNotIn(sentinel.encode(), file.read_bytes())
                    self.assertNotIn(file.suffix, ['.csv', '.parquet'])
            from mlflow.telemetry import get_telemetry_client
            self.assertIsNone(get_telemetry_client())

    def test_failed_roundtrip_does_not_create_a_registry_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            estimator = pipeline(LogisticRegression(max_iter=1000))
            report = run(make_demo(240), root/'run', model_candidates={'lr': estimator})
            bundle = joblib.load(root/'run/model.joblib')
            with patch('churn.tracking.verify_predictions', side_effect=AssertionError('changed predictions')):
                with self.assertRaises(AssertionError):
                    log_training(report, bundle, {'lr': estimator}, root/'run', root/'tracking')
            client, experiment_id = configure(root/'tracking')
            self.assertFalse(client.search_registered_models())
            self.assertIn('FAILED', [r.info.status for r in client.search_runs([experiment_id])])
            self.assertFalse((root/'run/mlflow.json').exists())

    def test_existing_experiment_cannot_redirect_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client, _ = configure(root/'tracking')
            client.create_experiment('outside', artifact_location=(root/'outside').as_uri())
            with self.assertRaisesRegex(ValueError, 'outside the tracking'):
                configure(root/'tracking', 'outside')


if __name__ == '__main__':
    unittest.main()
