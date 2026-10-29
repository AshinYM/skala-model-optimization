import tempfile
from pathlib import Path
import unittest

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from churn.demo import make_demo
from churn.features import Features, pipeline
from churn.metrics import contact_mask, evaluate
from churn.train import run, score, validate_training


class AuditEstimator(ClassifierMixin, BaseEstimator):
    calls = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.training_ids_ = set(X.CustomerId)
        type(self).calls.append(('fit', set(X.CustomerId)))
        return self

    def predict_proba(self, X):
        type(self).calls.append(('predict', set(X.CustomerId)))
        p = 1 / (1 + np.exp(-(X.Age.to_numpy() - 50) / 10))
        return np.column_stack([1 - p, p])


class ChurnTests(unittest.TestCase):
    def test_personal_identifiers_label_and_extra_columns_never_enter_features(self):
        data = make_demo(120)
        data['future_outcome'] = data.Exited
        features = Features().fit_transform(data)
        self.assertFalse({'Exited', 'CustomerId', 'Surname', 'future_outcome'} & set(features))
        self.assertIn('accountDuration_days', features)

    def test_invalid_dates_age_and_unknown_category_are_handled(self):
        data = make_demo(120)
        model = pipeline(LogisticRegression(max_iter=1000)).fit(data, data.Exited)
        incoming = data.iloc[:3].copy()
        incoming.loc[incoming.index[0], 'Age'] = 0
        incoming.loc[incoming.index[1], 'accountOpeningDate'] = '2099-01-01'
        incoming.loc[incoming.index[2], 'Geography'] = 'never-seen'
        self.assertTrue(np.isfinite(model.predict_proba(incoming)).all())
        features = model.named_steps['features'].transform(incoming)
        self.assertTrue(np.isnan(features.iloc[0].salaryPerAge))
        self.assertTrue(np.isnan(features.iloc[1].accountDuration_days))

    def test_missing_schema_and_infinite_values_fail_clearly(self):
        data = make_demo(120)
        with self.assertRaisesRegex(ValueError, 'Missing input columns'):
            Features().transform(data.drop(columns='Balance'))
        data['Balance'] = np.inf
        with self.assertRaisesRegex(ValueError, 'Infinite input'):
            Features().transform(data)

    def test_imputer_does_not_learn_from_inference(self):
        data = make_demo(120)
        model = pipeline(LogisticRegression(max_iter=1000)).fit(data, data.Exited)
        imputer = model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
        before = imputer.statistics_.copy()
        incoming = data.iloc[:3].copy()
        incoming['EstimatedSalary'] = 1e12
        model.predict_proba(incoming)
        np.testing.assert_array_equal(before, imputer.statistics_)

    def test_raw_input_predictions_survive_save_load_for_both_feature_variants(self):
        data = make_demo(120)
        with tempfile.TemporaryDirectory() as directory:
            for engineered in [False, True]:
                model = pipeline(LogisticRegression(max_iter=1000), engineered).fit(data, data.Exited)
                path = Path(directory)/'model.joblib'
                joblib.dump(model, path)
                restored = joblib.load(path)
                np.testing.assert_allclose(model.predict_proba(data), restored.predict_proba(data.drop(columns='Exited')))

    def test_budget_is_exact_and_ties_are_deterministic(self):
        selected = contact_mask(np.full(23, .5), .1)
        self.assertEqual(np.flatnonzero(selected).tolist(), [0, 1])
        self.assertEqual(contact_mask([.8, .2], .1).sum(), 0)
        metrics = evaluate([1, 0, 1, 0], [.9, .8, .7, .1], fraction=.5)
        self.assertEqual(metrics['captured_churners'], 1)
        self.assertEqual(metrics['lift_at_budget'], 1)
        for invalid in [[np.nan], [1.1], []]:
            with self.assertRaises(ValueError):
                contact_mask(invalid)
        self.assertIsNone(evaluate([0, 0], [.2, .1], fraction=.5)['lift_at_budget'])

    def test_split_isolation_and_only_final_model_predicts_test(self):
        data = make_demo(200)
        AuditEstimator.calls = []
        with tempfile.TemporaryDirectory() as directory:
            report = run(data, Path(directory)/'run', model_candidates={'audit': AuditEstimator()})
            calls = AuditEstimator.calls
            self.assertEqual([kind for kind, _ in calls], ['fit', 'predict', 'fit', 'predict'])
            train, validation, refit, test = [ids for _, ids in calls]
            self.assertFalse(train & validation or train & test or validation & test)
            self.assertEqual(refit, train | validation)
            self.assertEqual(len(train | validation | test), len(data))
            self.assertNotIn('test', report['validation_candidates'][0])
            bundle = joblib.load(Path(directory)/'run/model.joblib')
            scored = score(data.drop(columns='Exited'), bundle)
            self.assertEqual(scored.contact_selected.sum(), 20)
            self.assertNotIn('Surname', scored)

    def test_duplicate_customer_and_label_errors_are_rejected(self):
        data = make_demo(120)
        data.loc[1, 'CustomerId'] = data.loc[0, 'CustomerId']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_training(data)
        data = make_demo(120)
        data['Exited'] = 0
        with self.assertRaisesRegex(ValueError, 'both 0 and 1'):
            validate_training(data)


if __name__ == '__main__':
    unittest.main()
