import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.demo import make_demo
from churn.features import pipeline

API_AVAILABLE = all(importlib.util.find_spec(name) for name in ['fastapi', 'httpx'])
if API_AVAILABLE:
    from fastapi.testclient import TestClient
    from churn.api import create_app, MAX_BODY_BYTES


class ConstantModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.tile([.4, .6], (len(frame), 1))


@unittest.skipUnless(API_AVAILABLE, 'Install requirements-api.txt for API tests')
class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = make_demo(240)
        cls.models = {value: pipeline(LogisticRegression(max_iter=1000), value).fit(cls.data, cls.data.Exited)
                      for value in [False, True]}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/'model.joblib'
        self.bundle = {'pipeline': ConstantModel(), 'threshold': .7, 'contact_fraction': .1,
                       'model': 'test_model', 'seed': 42, 'schema_version': 1}
        joblib.dump(self.bundle, self.path)
        self.customers = self.data.drop(columns=['Surname', 'Exited']).iloc[:23].to_dict(orient='records')

    def cohort(self, customers=None):
        values = self.customers if customers is None else customers
        return {'cohort_id': 'synthetic-cohort', 'expected_customer_count': len(values), 'customers': values}

    def test_liveness_readiness_and_missing_model(self):
        with TestClient(create_app(Path(self.temp.name)/'missing.joblib')) as client:
            self.assertEqual(client.get('/health').json(), {'status': 'ok'})
            for response in [client.get('/ready'), client.post('/api/v1/scores', json={'customers': self.customers})]:
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json()['error']['code'], 'MODEL_UNAVAILABLE')
                self.assertNotIn(self.temp.name, response.text)

    def test_corrupt_bundle_or_policy_remains_unready(self):
        cases = [b'not a joblib file', dict(self.bundle, schema_version=99),
                 dict(self.bundle, threshold=float('nan')), dict(self.bundle, contact_fraction=0)]
        for value in cases:
            if isinstance(value, bytes):
                self.path.write_bytes(value)
            else:
                joblib.dump(value, self.path)
            with TestClient(create_app(self.path)) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/ready').status_code, 503)

    def test_real_pipeline_scores_match_batch_and_single_requests(self):
        for engineered, model in self.models.items():
            joblib.dump(dict(self.bundle, pipeline=model), self.path)
            with TestClient(create_app(self.path)) as client:
                response = client.post('/api/v1/scores', json={'customers': self.customers})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                expected = model.predict_proba(pd.DataFrame(self.customers))[:, 1]
                np.testing.assert_allclose([row['churn_score'] for row in result['scores']], expected, atol=1e-12)
                one = client.post('/api/v1/scores', json={'customers': self.customers[:1]}).json()
                self.assertAlmostEqual(one['scores'][0]['churn_score'], expected[0], places=12)
                self.assertEqual(result['model']['model_version'], 'sha256:'+hashlib.sha256(self.path.read_bytes()).hexdigest())
                self.assertNotIn('contact_selected', response.text)
                self.assertNotIn('rank', response.text)

    def test_complete_cohort_budget_and_order_independent_ties(self):
        with TestClient(create_app(self.path)) as client:
            first = client.post('/api/v1/contact-selections', json=self.cohort()).json()
            second = client.post('/api/v1/contact-selections', json=self.cohort(self.customers[::-1])).json()
            for result in [first, second]:
                selected = [r['CustomerId'] for r in result['selections'] if r['contact_selected']]
                self.assertEqual(set(selected), {1, 2})
                self.assertEqual(result['selected_count'], 2)
                self.assertEqual(sorted(r['rank'] for r in result['selections']), list(range(1, 24)))
                self.assertFalse(any(r['risk_flag'] for r in result['selections']))
            self.assertEqual([r['CustomerId'] for r in second['selections']], [c['CustomerId'] for c in self.customers[::-1]])
            small = client.post('/api/v1/contact-selections', json=self.cohort(self.customers[:9])).json()
            self.assertEqual(small['selected_count'], 0)

    def test_partial_declared_cohort_and_duplicates_are_rejected(self):
        with TestClient(create_app(self.path)) as client:
            partial = self.cohort(self.customers[:5])
            partial['expected_customer_count'] = 23
            response = client.post('/api/v1/contact-selections', json=partial)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()['error']['details'][0]['type'], 'cohort_count_mismatch')
            duplicate = [self.customers[0], self.customers[0]]
            for path, payload in [('/api/v1/scores', {'customers': duplicate}),
                                  ('/api/v1/contact-selections', self.cohort(duplicate))]:
                response = client.post(path, json=payload)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()['error']['details'][0]['type'], 'duplicate_customer_id')

    def test_schema_errors_do_not_echo_customer_values(self):
        sentinel = 'PRIVATE_VALUE_DO_NOT_ECHO'
        cases = [{'Age': '40'}, {'Age': 0}, {'HasCrCard': True}, {'baseDate': 'bad-date'},
                 {'accountOpeningDate': '2099-01-01'}, {'Surname': sentinel},
                 {'Exited': 1}, {'Age': sentinel}, {sentinel: sentinel}]
        with TestClient(create_app(self.path)) as client:
            for update in cases:
                customer = dict(self.customers[0], **update)
                response = client.post('/api/v1/scores', json={'customers': [customer]})
                self.assertEqual(response.status_code, 422, str(update))
                self.assertNotIn(sentinel, response.text)
                self.assertNotIn('input', response.json()['error']['details'][0])

    def test_nullable_features_and_new_categories_are_supported(self):
        joblib.dump(dict(self.bundle, pipeline=self.models[True]), self.path)
        customer = dict(self.customers[0], Geography='never-seen', Gender=None,
                        EstimatedSalary=None, CreditScore=None, baseDate=None, accountOpeningDate=None)
        with TestClient(create_app(self.path)) as client:
            result = client.post('/api/v1/scores', json={'customers': [customer]})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertTrue(np.isfinite(result.json()['scores'][0]['churn_score']))

    def test_row_limits_and_empty_requests(self):
        with TestClient(create_app(self.path)) as client:
            for customers in [[], [dict(self.customers[0], CustomerId=i+1) for i in range(1001)]]:
                self.assertEqual(client.post('/api/v1/scores', json={'customers': customers}).status_code, 422)
            values = [dict(self.customers[0], CustomerId=i+1) for i in range(5001)]
            self.assertEqual(client.post('/api/v1/contact-selections', json=self.cohort(values)).status_code, 422)
            payload = self.cohort()
            payload['contact_fraction'] = .5
            self.assertEqual(client.post('/api/v1/contact-selections', json=payload).status_code, 422)

    def test_body_size_is_bounded_for_normal_and_chunked_bodies(self):
        with TestClient(create_app(self.path)) as client:
            for body in [b' '*(MAX_BODY_BYTES+1), iter([b' '*MAX_BODY_BYTES, b' '])]:
                response = client.post('/api/v1/scores', content=body, headers={'content-type': 'application/json'})
                self.assertEqual(response.status_code, 413)
                self.assertEqual(response.json()['error']['code'], 'BODY_TOO_LARGE')

    def test_invalid_json_and_nonfinite_values_are_sanitized(self):
        with TestClient(create_app(self.path)) as client:
            for body in ['{"customers": [PRIVATE_VALUE_DO_NOT_ECHO',
                         json.dumps({'customers': [dict(self.customers[0], Balance=float('nan'))]})]:
                response = client.post('/api/v1/scores', content=body, headers={'content-type': 'application/json'})
                self.assertEqual(response.status_code, 422)
                self.assertNotIn('PRIVATE_VALUE_DO_NOT_ECHO', response.text)

    def test_inference_failures_are_safe_and_do_not_look_successful(self):
        app = create_app(self.path)
        with TestClient(app) as client:
            for replacement in [ValueError('PRIVATE_PATH_DO_NOT_ECHO'), np.array([[np.nan, .5]]), np.array([[.8, .8]])]:
                kwargs = {'side_effect': replacement} if isinstance(replacement, Exception) else {'return_value': replacement}
                with patch.object(app.state.loaded.pipeline, 'predict_proba', **kwargs):
                    response = client.post('/api/v1/scores', json={'customers': self.customers[:1]})
                    self.assertEqual(response.status_code, 500)
                    self.assertEqual(response.json()['error']['code'], 'PREDICTION_FAILED')
                    self.assertNotIn('PRIVATE_PATH_DO_NOT_ECHO', response.text)

    def test_model_is_loaded_once_and_file_changes_require_restart(self):
        from churn.api import load_model
        with patch('churn.api.load_model', wraps=load_model) as loader:
            with TestClient(create_app(self.path)) as client:
                before = client.get('/ready').json()['model']
                joblib.dump(dict(self.bundle, threshold=.2), self.path)
                client.post('/api/v1/scores', json={'customers': self.customers})
                self.assertEqual(client.get('/ready').json()['model'], before)
                self.assertEqual(loader.call_count, 1)
        with TestClient(create_app(self.path)) as client:
            self.assertNotEqual(client.get('/ready').json()['model']['model_version'], before['model_version'])

    def test_openapi_is_checked_in_and_errors_are_documented(self):
        app = create_app(self.path)
        expected = json.loads((Path(__file__).resolve().parents[1]/'api/openapi.json').read_text())
        self.assertEqual(app.openapi(), expected)
        self.assertNotIn('Surname', app.openapi()['components']['schemas']['Customer']['properties'])
        self.assertEqual(app.openapi()['paths']['/api/v1/scores']['post']['responses']['422']['content']['application/json']['schema'],
                         {'$ref': '#/components/schemas/ErrorResponse'})


if __name__ == '__main__':
    unittest.main()
