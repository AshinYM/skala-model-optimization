"""Explicit local MLflow logging; no customer samples or automatic promotion."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import unquote, urlparse

import joblib
import numpy as np

from .demo import make_demo
from .features import NUMERIC, CATEGORICAL, DATES


def local_artifacts(uri, root):
    parsed = urlparse(uri)
    if parsed.scheme != 'file' or parsed.netloc:
        raise ValueError('This workflow only accepts local file artifacts')
    path = Path(unquote(parsed.path)).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Experiment artifact location is outside the tracking directory')
    return path


def configure(directory, experiment='bank-churn'):
    # Set these before MLflow is imported, including in CLI subprocesses.
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['MLFLOW_ENABLE_OTLP_EXPORTER'] = 'false'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = '1'
    import mlflow
    from mlflow.tracking import MlflowClient
    if mlflow.active_run() is not None:
        raise ValueError('Finish the active MLflow run before starting this workflow')
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    uri = 'sqlite:///' + (root/'tracking.db').as_posix()
    # Ignore inherited remote tracking/registry endpoints deliberately.
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        experiment_id = client.create_experiment(experiment, artifact_location=(root/'artifacts').as_uri())
    else:
        if found.lifecycle_stage != 'active':
            raise ValueError('Experiment is deleted; choose a new experiment name')
        local_artifacts(found.artifact_location, root)
        experiment_id = found.experiment_id
    return client, experiment_id


def probe():
    data = make_demo(rows=8, seed=2026)[NUMERIC + CATEGORICAL + DATES].copy()
    data[NUMERIC] = data[NUMERIC].astype(float)
    return data


def scalar_metrics(values, prefix):
    return {prefix + key: float(value) for key, value in values.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)}


def estimator_params(estimator):
    result = {}
    for name, value in estimator.get_params(deep=True).items():
        if value is None or isinstance(value, (str, bool, int, float)):
            result[name] = str(value) if isinstance(value, float) and not math.isfinite(value) else value
    return result


def verify_predictions(model, reference, inputs):
    actual = model.predict_proba(inputs)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12)
    return float(np.max(np.abs(actual - reference)))


def log_training(report, bundle, estimators, output, directory, experiment='bank-churn'):
    client, experiment_id = configure(directory, experiment)
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    output = Path(output)
    features = probe()
    expected = bundle['pipeline'].predict_proba(features)
    policy = {key: bundle[key] for key in ['model', 'threshold', 'contact_fraction', 'seed', 'schema_version']}
    validation = {row['model']: row for row in report['validation_candidates']}
    candidate_ids = []
    with mlflow.start_run(experiment_id=experiment_id, run_name='selected-final',
                          tags={'phase': 'selected_final', 'selection_split': 'validation',
                                'deployment': 'not_deployed'}, log_system_metrics=False) as parent:
        mlflow.log_params({'seed': report['seed'], 'selected_model': bundle['model'],
            'dataset_sha256': report['dataset_sha256'] or 'not_provided',
            'contact_fraction': bundle['contact_fraction'], 'threshold_f2': bundle['threshold'],
            'train_rows': report['split']['train'], 'validation_rows': report['split']['validation'],
            'test_rows': report['split']['test'], 'split_strategy': report['split']['strategy']})
        mlflow.set_tags({'selection_rule': report['selection_rule'], 'threshold_rule': report['threshold_rule']})
        for name, estimator in estimators.items():
            with mlflow.start_run(experiment_id=experiment_id, run_name=name, nested=True,
                                  tags={'phase': 'candidate', 'evaluation_split': 'validation'},
                                  log_system_metrics=False) as child:
                candidate_ids.append(child.info.run_id)
                mlflow.log_params({'candidate': name, **estimator_params(estimator)})
                mlflow.log_metrics(scalar_metrics(validation[name], 'validation_'))
        mlflow.log_metrics(scalar_metrics(report['selected_validation'], 'selected_validation_'))
        mlflow.log_metrics(scalar_metrics(report['final_test'], 'test_'))
        mlflow.log_dict(report, 'report.json')
        mlflow.log_dict(policy, 'decision_policy.json')
        requirements = [f'{name}=={importlib.metadata.version(name)}'
                        for name in ['mlflow', 'cloudpickle', 'numpy', 'pandas', 'scikit-learn', 'joblib', 'xgboost']]
        # Bundle only application code, not caches, data, environment files or the repo.
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)/'churn'
            source.mkdir()
            for module in Path(__file__).parent.glob('*.py'):
                shutil.copyfile(module, source/module.name)
            logged = mlflow.sklearn.log_model(
                sk_model=bundle['pipeline'], name='selected_pipeline', code_paths=[str(source)],
                serialization_format='cloudpickle',
                signature=infer_signature(features, expected), input_example=None,
                pip_requirements=requirements, pyfunc_predict_fn='predict_proba', metadata=policy)
        artifact_error = verify_predictions(mlflow.sklearn.load_model(logged.model_uri), expected, features)
        # Register only after round-trip verification; do not set a production alias.
        version = mlflow.register_model(logged.model_uri, 'BankChurnPipeline', await_registration_for=30)
        registry_uri = f'models:/BankChurnPipeline/{version.version}'
        registry_error = verify_predictions(mlflow.sklearn.load_model(registry_uri), expected, features)
        client.set_model_version_tag('BankChurnPipeline', version.version, 'roundtrip_verified', 'true')
        client.set_model_version_tag('BankChurnPipeline', version.version, 'deployment', 'not_deployed')
        mlflow.log_metrics({'roundtrip_artifact_max_abs_error': artifact_error,
                            'roundtrip_registry_max_abs_error': registry_error})
        summary = {'schema_version': 1, 'run_id': parent.info.run_id, 'experiment_id': experiment_id,
            'candidate_run_ids': candidate_ids, 'model_uri': logged.model_uri,
            'registered_model': 'BankChurnPipeline', 'registered_version': str(version.version),
            'probe': '8 independently generated synthetic rows; no customer records',
            'artifact_max_abs_error': artifact_error, 'registry_max_abs_error': registry_error,
            'mlflow_version': importlib.metadata.version('mlflow'), 'deployment': 'not_deployed'}
        mlflow.log_dict(summary, 'roundtrip.json')
        (output/'mlflow.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def verify_saved(output, directory, experiment='bank-churn'):
    configure(directory, experiment)
    import mlflow.sklearn
    output = Path(output)
    metadata = json.loads((output/'mlflow.json').read_text())
    # Use a fixed local registry version, not an arbitrary URI from a file.
    version = str(metadata['registered_version'])
    if not version.isdigit():
        raise ValueError('Invalid registry version')
    original = joblib.load(output/'model.joblib')['pipeline']
    inputs = probe()
    restored = mlflow.sklearn.load_model(f'models:/BankChurnPipeline/{version}')
    return {'probe_rows': len(inputs), 'max_abs_error': verify_predictions(restored, original.predict_proba(inputs), inputs),
            'registered_version': version}


def main():
    parser = argparse.ArgumentParser(description='Verify a trusted local churn model against its MLflow registry version')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--tracking-dir', type=Path, required=True)
    parser.add_argument('--experiment', default='bank-churn')
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.run_dir, args.tracking_dir, args.experiment), indent=2))


if __name__ == '__main__':
    main()
