"""Local-only churn inference API. No model uploads, customer storage or remote calls."""
import argparse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import date
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
from typing import Annotated, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError
from starlette.exceptions import HTTPException

from .demo import make_demo
from .features import NUMERIC, CATEGORICAL, DATES
from .metrics import contact_mask

MAX_BODY_BYTES = 2 * 1024 * 1024
Number = Annotated[float, Field(strict=True)]
Identifier = Annotated[int, Field(strict=True, ge=1)]
Flag = Annotated[int, Field(strict=True, ge=0, le=1)]
Category = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Customer(StrictModel):
    CustomerId: Identifier
    CreditScore: Number | None
    Age: Annotated[float, Field(strict=True, gt=0)] | None
    Balance: Number | None
    NumOfProducts: Annotated[int, Field(strict=True, ge=0)] | None
    HasCrCard: Flag | None
    IsActiveMember: Flag | None
    EstimatedSalary: Number | None
    Geography: Category | None
    Gender: Category | None
    baseDate: date | None
    accountOpeningDate: date | None

    @model_validator(mode='after')
    def valid_dates(self):
        if self.baseDate and self.accountOpeningDate and self.accountOpeningDate > self.baseDate:
            raise PydanticCustomError('opening_date_after_base', 'Opening date must not follow base date')
        return self


def unique_customers(customers):
    if len({c.CustomerId for c in customers}) != len(customers):
        raise PydanticCustomError('duplicate_customer_id', 'Customer IDs must be unique')


class ScoresRequest(StrictModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)

    @model_validator(mode='after')
    def unique(self):
        unique_customers(self.customers)
        return self


class CohortRequest(StrictModel):
    cohort_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    expected_customer_count: Annotated[int, Field(strict=True, ge=1, le=5000)]
    customers: list[Customer] = Field(min_length=1, max_length=5000)

    @model_validator(mode='after')
    def complete_and_unique(self):
        unique_customers(self.customers)
        if len(self.customers) != self.expected_customer_count:
            raise PydanticCustomError('cohort_count_mismatch', 'Submit the complete declared cohort')
        return self


class ModelInfo(StrictModel):
    model_name: str
    model_version: str
    schema_version: int
    threshold: float
    contact_fraction: float


class ScoreRow(StrictModel):
    CustomerId: int
    churn_score: float = Field(ge=0, le=1)
    risk_flag: bool


class ScoresResponse(StrictModel):
    model: ModelInfo
    count: int
    scores: list[ScoreRow]


class SelectionRow(ScoreRow):
    rank: int = Field(ge=1)
    contact_selected: bool


class SelectionResponse(StrictModel):
    model: ModelInfo
    cohort_id: str
    cohort_size: int
    selected_count: int
    tie_breaker: Literal['CustomerId ascending']
    selection_scope: Literal['complete declared cohort in this request']
    selections: list[SelectionRow]


class ErrorDetail(StrictModel):
    field: list[str | int]
    type: str


class ErrorBody(StrictModel):
    code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(StrictModel):
    error: ErrorBody


class ReadyResponse(StrictModel):
    status: Literal['ready']
    model: ModelInfo


class HealthResponse(StrictModel):
    status: Literal['ok']


def error(status, code, message, details=None):
    return JSONResponse(status_code=status, content={
        'error': {'code': code, 'message': message, 'details': details or []}})


class APIError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


class BodyLimit:
    """Bound bytes actually received as well as Content-Length, before JSON parsing."""
    def __init__(self, app, limit=MAX_BODY_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        length = dict(scope.get('headers', [])).get(b'content-length', b'0')
        if not length.isdigit():
            return await error(400, 'INVALID_CONTENT_LENGTH', 'Invalid Content-Length')(scope, receive, send)
        if int(length) > self.limit:
            return await error(413, 'BODY_TOO_LARGE', 'Request body exceeds 2 MiB')(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            chunk = message.get('body', b'')
            size += len(chunk)
            if size > self.limit:
                return await error(413, 'BODY_TOO_LARGE', 'Request body exceeds 2 MiB')(scope, receive, send)
            chunks.append(chunk)
            if not message.get('more_body', False):
                break
        body, delivered = b''.join(chunks), False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {'type': 'http.request', 'body': body, 'more_body': False}
            return await receive()

        await self.app(scope, replay, send)


def valid_probabilities(value, rows):
    probabilities = np.asarray(value)
    if (probabilities.shape != (rows, 2) or not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
            or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError('Invalid classifier output')
    return probabilities[:, 1]


@dataclass
class LoadedModel:
    pipeline: object
    info: ModelInfo
    lock: Lock = field(default_factory=Lock)

    def predict(self, customers):
        rows = [c.model_dump(mode='json', exclude={'CustomerId'}) for c in customers]
        frame = pd.DataFrame(rows)
        # Explicit nulls become numeric missing values, matching the training Pipeline.
        frame[NUMERIC] = frame[NUMERIC].astype(float)
        try:
            with self.lock:
                return valid_probabilities(self.pipeline.predict_proba(frame), len(rows))
        except Exception:
            raise APIError(500, 'PREDICTION_FAILED', 'Model inference failed') from None


def load_model(path):
    if not path:
        raise ValueError('No model configured')
    raw = Path(path).read_bytes()
    bundle = joblib.load(io.BytesIO(raw))
    if not isinstance(bundle, dict) or bundle.get('schema_version') != 1:
        raise ValueError('Unsupported model bundle')
    name = bundle.get('model')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', name):
        raise ValueError('Invalid model name')
    threshold, fraction = bundle['threshold'], bundle['contact_fraction']
    if (isinstance(threshold, bool) or isinstance(fraction, bool)
            or not math.isfinite(threshold) or not 0 <= threshold <= 1
            or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError('Invalid decision policy')
    model = bundle['pipeline']
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError('Expected class order [0, 1]')
    sample = make_demo(8)[NUMERIC + CATEGORICAL + DATES]
    valid_probabilities(model.predict_proba(sample), len(sample))
    return LoadedModel(model, ModelInfo(model_name=name, model_version='sha256:'+hashlib.sha256(raw).hexdigest(),
        schema_version=1, threshold=threshold, contact_fraction=fraction))


def create_app(model_path=None):
    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.loaded = load_model(model_path or os.getenv('CHURN_MODEL_PATH'))
        except Exception:
            # Remain live but unready; never expose the path, artifact or exception.
            app.state.loaded = None
        yield

    app = FastAPI(title='Bank Churn Scoring API', version='1.0.0', lifespan=lifespan,
        description='Local-only scoring and complete-declared-cohort selection. No customer storage, model upload or automatic deployment.')
    app.add_middleware(BodyLimit)

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        allowed = set(Customer.model_fields) | {'body', 'customers', 'cohort_id', 'expected_customer_count'}
        details = [{'field': [x if isinstance(x, int) or x in allowed else 'unknown_field' for x in e['loc']],
                    'type': e['type']} for e in exc.errors()]
        return error(422, 'INVALID_REQUEST', 'Request validation failed', details)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return error(exc.status_code, 'HTTP_ERROR', 'Request could not be handled')

    def ready():
        loaded = app.state.loaded
        if loaded is None:
            raise APIError(503, 'MODEL_UNAVAILABLE', 'Model is unavailable; configure a trusted bundle and restart')
        return loaded

    @app.get('/health', response_model=HealthResponse)
    def health():
        return {'status': 'ok'}

    @app.get('/ready', response_model=ReadyResponse, responses={503: {'model': ErrorResponse}})
    def readiness():
        return {'status': 'ready', 'model': ready().info}

    failures = {code: {'model': ErrorResponse} for code in [400, 413, 422, 500, 503]}

    @app.post('/api/v1/scores', response_model=ScoresResponse, responses=failures,
              description='Score any subset. Does not select contact targets or allocate a contact budget.')
    def scores(request: ScoresRequest):
        loaded = ready()
        p = loaded.predict(request.customers)
        return {'model': loaded.info, 'count': len(p), 'scores': [
            {'CustomerId': c.CustomerId, 'churn_score': float(value), 'risk_flag': bool(value >= loaded.info.threshold)}
            for c, value in zip(request.customers, p)]}

    @app.post('/api/v1/contact-selections', response_model=SelectionResponse, responses=failures,
              description='Submit one complete declared cohort, not chunks. Expected count must match. No cross-request aggregation or storage.')
    def selections(request: CohortRequest):
        loaded = ready()
        p = loaded.predict(request.customers)
        ids = np.array([c.CustomerId for c in request.customers])
        canonical = np.argsort(ids, kind='stable')
        selected = np.zeros(len(ids), dtype=bool)
        selected[canonical] = contact_mask(p[canonical], loaded.info.contact_fraction)
        ordering = np.lexsort((ids, -p))
        ranks = np.empty(len(ids), dtype=int)
        ranks[ordering] = np.arange(1, len(ids)+1)
        return {'model': loaded.info, 'cohort_id': request.cohort_id, 'cohort_size': len(ids),
            'selected_count': int(selected.sum()), 'tie_breaker': 'CustomerId ascending',
            'selection_scope': 'complete declared cohort in this request',
            'selections': [{'CustomerId': c.CustomerId, 'churn_score': float(p[i]),
                'risk_flag': bool(p[i] >= loaded.info.threshold), 'rank': int(ranks[i]),
                'contact_selected': bool(selected[i])} for i, c in enumerate(request.customers)]}

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-openapi', type=Path, required=True)
    args = parser.parse_args()
    args.export_openapi.parent.mkdir(parents=True, exist_ok=True)
    args.export_openapi.write_text(json.dumps(app.openapi(), ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
