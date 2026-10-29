"""Generate fictional data for installation checks, never a bank benchmark."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_demo(rows=1000, seed=2026):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, rows)
    products = rng.integers(1, 4, rows)
    active = rng.integers(0, 2, rows)
    p = 1 / (1 + np.exp(-(-3 + .045 * (age - 30) + .8 * (products == 1) - .7 * active)))
    return pd.DataFrame({'CustomerId': np.arange(1, rows + 1), 'Surname': 'SYNTHETIC',
        'CreditScore': rng.integers(400, 850, rows), 'Geography': rng.choice(['A', 'B', 'C'], rows),
        'Gender': rng.choice(['A', 'B'], rows), 'Age': age, 'Balance': rng.uniform(0, 200000, rows),
        'NumOfProducts': products, 'HasCrCard': rng.integers(0, 2, rows),
        'IsActiveMember': active, 'EstimatedSalary': rng.uniform(20000, 150000, rows),
        'baseDate': '2026-01-01',
        'accountOpeningDate': (pd.Timestamp('2026-01-01') - pd.to_timedelta(rng.integers(30, 3000, rows), unit='D')).astype(str),
        'Exited': rng.binomial(1, p)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--rows', default=1000, type=int)
    args = parser.parse_args()
    if args.rows < 100:
        parser.error('Use at least 100 rows')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        make_demo(args.rows).to_csv(handle, index=False)
    print('Synthetic data generated; metrics on it are not real-world evidence')


if __name__ == '__main__':
    main()
