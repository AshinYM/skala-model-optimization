"""Check a local API against a trusted local bundle, using synthetic customers only."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd

from .demo import make_demo


def verify(url, model_path):
    parsed = urlparse(url)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'} or parsed.username:
        raise ValueError('Use an HTTP localhost target only')
    base = url.rstrip('/')

    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base+path, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urlopen(req, timeout=20) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    customers = make_demo(23).drop(columns=['Surname', 'Exited']).to_dict(orient='records')
    raw = Path(model_path).read_bytes()
    bundle = joblib.load(model_path)
    checks = {}
    checks['liveness'] = request('/health') == (200, {'status': 'ok'})
    status, ready = request('/ready')
    checks['readiness'] = status == 200
    version = 'sha256:'+hashlib.sha256(raw).hexdigest()
    checks['model_version_matches_file'] = ready['model']['model_version'] == version
    status, result = request('/api/v1/scores', {'customers': customers})
    expected = bundle['pipeline'].predict_proba(pd.DataFrame(customers))[:, 1]
    actual = np.array([row['churn_score'] for row in result['scores']])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    checks['scores_match_local_pipeline'] = status == 200
    checks['score_operation_does_not_select_contacts'] = all('contact_selected' not in row and 'rank' not in row for row in result['scores'])
    checks['threshold_policy_matches'] = all(row['risk_flag'] == bool(p >= bundle['threshold'])
                                            for row, p in zip(result['scores'], expected))
    cohort = {'cohort_id': 'synthetic-check', 'expected_customer_count': len(customers), 'customers': customers}
    status, selected = request('/api/v1/contact-selections', cohort)
    count = int(np.floor(len(customers)*bundle['contact_fraction']))
    checks['cohort_budget_respected'] = status == 200 and selected['selected_count'] == count
    actual_selected = {row['CustomerId'] for row in selected['selections'] if row['contact_selected']}
    ids = np.array([row['CustomerId'] for row in customers])
    expected_selected = set(ids[np.lexsort((ids, -expected))[:count]])
    checks['global_rank_selection_matches'] = actual_selected == expected_selected
    partial = dict(cohort, customers=customers[:5])
    checks['partial_declared_cohort_rejected'] = request('/api/v1/contact-selections', partial)[0] == 422
    sentinel = 'PRIVATE_VALUE_DO_NOT_ECHO'
    status, failure = request('/api/v1/scores', {'customers': [dict(customers[0], Surname=sentinel)]})
    checks['invalid_input_sanitized'] = status == 422 and sentinel not in json.dumps(failure)
    status, spec = request('/openapi.json')
    checks['openapi_available'] = status == 200 and '/api/v1/scores' in spec['paths']
    assert all(checks.values()), 'API verification failed: '+', '.join(k for k, v in checks.items() if not v)
    return {'checks': checks, 'checks_passed': len(checks), 'synthetic_customers': len(customers),
            'model_name': ready['model']['model_name'], 'max_abs_score_error': float(np.max(np.abs(actual-expected))),
            'customer_records_saved': False, 'external_services_called': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8061')
    parser.add_argument('--model', required=True, type=Path, help='Your own trusted local bundle only')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = verify(args.url, args.model)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            handle.write(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
