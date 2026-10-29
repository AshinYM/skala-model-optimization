import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .features import pipeline
from .metrics import bootstrap, choose_threshold, contact_mask, evaluate


def validate_training(data):
    if not {'Exited', 'CustomerId'} <= set(data.columns):
        raise ValueError('Exited and CustomerId are required')
    if data.CustomerId.isna().any() or data.CustomerId.duplicated().any():
        raise ValueError('One snapshot per customer is required; missing/duplicate CustomerId')
    if data.Exited.isna().any() or set(data.Exited.unique()) != {0, 1}:
        raise ValueError('Exited must contain both 0 and 1, with no missing labels')
    if data.Exited.value_counts().min() < 10 or len(data) < 100:
        raise ValueError('Need at least 100 rows and 10 examples of each label')


def candidates(seed):
    rf = dict(n_estimators=240, max_depth=10, min_samples_leaf=3,
              max_features='sqrt', class_weight='balanced', random_state=seed, n_jobs=2)
    return {
        'logistic_fe': pipeline(LogisticRegression(max_iter=2000, random_state=seed)),
        'rf_baseline': pipeline(RandomForestClassifier(**rf), engineered=False),
        'rf_fe': pipeline(RandomForestClassifier(**rf)),
        'rf_shallow_fe': pipeline(RandomForestClassifier(**(rf | {'max_depth': 6, 'min_samples_leaf': 5}))),
        'xgboost_fe': pipeline(XGBClassifier(n_estimators=300, max_depth=3, learning_rate=.04,
            subsample=.85, colsample_bytree=.85, min_child_weight=3, reg_lambda=2,
            eval_metric='logloss', random_state=seed, n_jobs=2)),
    }


def run(data, output, seed=42, fraction=0.1, model_candidates=None, dataset_sha256=None,
        tracking_dir=None, experiment='bank-churn'):
    validate_training(data)
    if not 0 < fraction <= 1:
        raise ValueError('Contact fraction must be in (0, 1]')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = data.reset_index(drop=True)
    train, hold = train_test_split(np.arange(len(data)), test_size=.4, stratify=data.Exited, random_state=seed)
    validation, test = train_test_split(hold, test_size=.5, stratify=data.Exited.iloc[hold], random_state=seed)
    if int(len(validation) * fraction) == 0 or int(len(test) * fraction) == 0:
        raise ValueError('Contact budget selects zero rows in validation/test')
    x = data.drop(columns='Exited')
    y = data.Exited.astype(int)
    models = model_candidates if model_candidates is not None else candidates(seed)
    if not models:
        raise ValueError('At least one model candidate is required')
    scores, fitted, validation_probabilities, audit = [], {}, {}, []
    for name, estimator in models.items():
        model = clone(estimator).fit(x.iloc[train], y.iloc[train])
        p = model.predict_proba(x.iloc[validation])[:, 1]
        result = evaluate(y.iloc[validation], p, fraction=fraction)
        scores.append({'model': name, **result})
        fitted[name], validation_probabilities[name] = model, p
        audit.append({'phase': 'candidate', 'model': name, 'fit': 'train', 'evaluate': 'validation'})
    # The held-out test data is not used in this ranking or threshold choice.
    ranking = sorted(scores, key=lambda r: (-r['precision_at_budget'], -r['average_precision'], -r['roc_auc'], r['model']))
    winner = ranking[0]['model']
    threshold = choose_threshold(y.iloc[validation], validation_probabilities[winner])
    validation_selected = evaluate(y.iloc[validation], validation_probabilities[winner], threshold, fraction)
    final = clone(fitted[winner]).fit(x.iloc[np.concatenate([train, validation])], y.iloc[np.concatenate([train, validation])])
    audit.append({'phase': 'refit', 'model': winner, 'fit': 'train+validation'})
    test_p = final.predict_proba(x.iloc[test])[:, 1]
    audit.append({'phase': 'final_evaluation', 'model': winner, 'evaluate': 'test', 'prediction_calls': 1})
    test_result = evaluate(y.iloc[test], test_p, threshold, fraction)
    report = {
        'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'dataset_sha256': dataset_sha256, 'seed': seed,
        'split': {'train': len(train), 'validation': len(validation), 'test': len(test),
                  'strategy': 'stratified 60/20/20, one snapshot per customer'},
        'selection_rule': 'validation precision_at_budget DESC, average_precision DESC, roc_auc DESC, name ASC',
        'threshold_rule': 'validation F2 maximum; ties choose highest threshold; frozen before final refit',
        'selected_model': winner, 'validation_candidates': ranking,
        'selected_validation': validation_selected, 'final_test': test_result,
        'test_bootstrap_95_percent': bootstrap(y.iloc[test], test_p, fraction, seed),
        'test_contact_budget_curve': [evaluate(y.iloc[test], test_p, threshold, f) for f in [.05, .1, .2]],
        'audit': audit,
        'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'xgboost']}},
        'limitations': [
            'Retrospective educational data already explored in prior notebooks; not a pristine external holdout.',
            'One split, small validation contact cohort; model selection can be unstable.',
            'Bootstrap intervals condition on fixed predictions and do not include training/selection uncertainty.',
            'Scores are not calibrated probabilities or causal retention effects.',
            'Budget curves are descriptive; do not retune the policy using test results.',
        ],
    }
    bundle = {'pipeline': final, 'threshold': threshold, 'contact_fraction': fraction,
              'model': winner, 'seed': seed, 'schema_version': 1}
    joblib.dump(bundle, output/'model.joblib')
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if tracking_dir is not None:
        from .tracking import log_training
        log_training(report, bundle, models, output, tracking_dir, experiment)
    return report


def score(data, bundle):
    if 'CustomerId' not in data or data.CustomerId.isna().any() or data.CustomerId.duplicated().any():
        raise ValueError('Scoring requires unique nonmissing CustomerId values')
    if data.empty:
        raise ValueError('No customers to score')
    p = bundle['pipeline'].predict_proba(data.drop(columns='Exited', errors='ignore'))[:, 1]
    return pd.DataFrame({'CustomerId': data.CustomerId.to_numpy(), 'churn_score': p,
        'risk_flag': p >= bundle['threshold'],
        'contact_selected': contact_mask(p, bundle['contact_fraction'])})


def main():
    parser = argparse.ArgumentParser(description='Bank churn training and batch scoring; no external API calls')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train')
    train.add_argument('--data', required=True, type=Path)
    train.add_argument('--output', required=True, type=Path, help='New directory; existing paths are never overwritten')
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--contact-fraction', type=float, default=.1)
    train.add_argument('--tracking-dir', type=Path, help='Opt-in local SQLite/artifact directory (no remote tracking)')
    train.add_argument('--experiment', default='bank-churn')
    predict = commands.add_parser('score')
    predict.add_argument('--data', required=True, type=Path)
    predict.add_argument('--model', required=True, type=Path, help='Only load your own trusted joblib artifact')
    predict.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'train':
        report = run(pd.read_csv(args.data), args.output, args.seed, args.contact_fraction,
                     dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                     tracking_dir=args.tracking_dir, experiment=args.experiment)
        print(json.dumps({'selected_model': report['selected_model'], 'final_test': report['final_test']}, indent=2))
    else:
        if args.output.exists():
            parser.error('Output file already exists')
        result = score(pd.read_csv(args.data), joblib.load(args.model))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive create avoids silently overwriting another run's customer scores.
        with args.output.open('x') as handle:
            result.to_csv(handle, index=False)
        print(f'Scored {len(result)} customers; individual scores saved locally')


if __name__ == '__main__':
    main()
