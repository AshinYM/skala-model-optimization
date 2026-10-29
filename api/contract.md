# 고객 이탈 API v1 계약

로컬 개인용 서비스. 인증·DB·외부 모델 호출·자동 재학습은 제공하지 않는다. 신뢰 가능한 로컬 joblib만 시작 시 한 번 불러온다. 공개 서버로 노출하지 않는다.

## 연산

| 경로 | 요청 | 성공 | 주요 실패 |
|---|---|---|---|
| GET /health | 없음 | 200: 프로세스 생존 | — |
| GET /ready | 없음 | 200: 모델 이름·파일 SHA-256 버전·정책 | 503: 모델 없음/손상/미지원 정책 |
| POST /api/v1/scores | customers 1~1,000명 | 200: 점수·risk_flag, 모델 버전·임계값 | 422: 잘못된 입력/중복 ID, 503: 모델 없음, 500: 추론 실패 |
| POST /api/v1/contact-selections | cohort_id, expected_customer_count, customers 1~5,000명 | 200: 전체 고객 점수·순위·연락 대상 여부·선정 수 | 422: 예상 수 불일치/중복/잘못된 입력, 503/500 |

두 POST 요청의 최대 본문 크기는 2 MiB이며 초과 시 413이다. 원본 고객 행·점수를 서버에 저장하거나 로그하지 않는다. 오류에는 입력 값·모델 파일 경로·예외 원문을 포함하지 않는다. 모든 오류는 `error.code`, `error.message`, `error.details` 형태다.

## 고객 입력

기존 CSV의 컬럼 이름을 사용한다. CustomerId는 1 이상의 정수이며 요청 안에서 유일해야 한다. 숫자로 된 문자열·bool을 숫자로 간주하지 않는다. 입력에 Exited·Surname·그 외 알려지지 않은 필드는 허용하지 않는다.

CreditScore, Age, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender, baseDate, accountOpeningDate를 모두 명시한다. 값이 없으면 null을 사용한다. 숫자는 유한해야 하고 Age는 양수, NumOfProducts는 0 이상의 정수, 두 플래그는 정수 0/1이다. 범주는 처음 보는 문자열도 허용한다. 날짜는 ISO 날짜이며 개설일이 기준일 뒤일 수 없다. 잘못된 날짜를 묵시적으로 결측값으로 바꾸는 CLI와 달리 API는 422로 거부한다.

## 점수와 연락 대상의 의미

scores는 부분 배치에도 사용할 수 있으며 연락 대상·순위 필드를 반환하지 않는다. 점수는 [0,1]의 모델 출력이며 실제 이탈 확률 보정·캠페인 효과를 보장하지 않는다. risk_flag는 저장된 Validation F2 임계값 기준이다.

contact-selections는 호출자가 정한 단일 전체 cohort를 한 번에 제출하는 연산이다. expected_customer_count는 customers 길이와 같아야 한다. 서버가 업무 DB의 실제 모집단을 대조하거나 여러 요청을 합치는 것은 아니다. cohort_id는 고객군의 식별용이며 저장/중복 방지/idempotency 키가 아니다. 5,000명을 넘는 모집단을 나눠서 이 연산에 보내지 않는다. 큰 cohort의 저장·분할 업로드는 향후 별도 설계가 필요하다.

선정 수는 floor(전체 고객 수 × 모델에 저장된 contact_fraction)이다. 요청으로 비율을 바꿀 수 없다. 위험 점수 내림차순, 동점이면 CustomerId 오름차순으로 순위를 정해 요청 행 순서가 바뀌어도 같은 고객이 선택된다. 결과 행은 입력 순서를 유지한다. CLI의 기존 동점 규칙(입력 순서)과는 구분한다. 선정 수가 0명인 작은 cohort도 그대로 반환한다.

model_version은 실제 불러온 파일의 SHA-256 전체 값이며 MLflow Registry 번호나 자동 배포 별칭이 아니다. 실행 중 모델 파일이 바뀌어도 재시작 전까지 같은 모델·정책·버전을 사용한다.

## 명세와 예제

필드별 기계 판독 명세는 FastAPI에서 생성한 `api/openapi.json`으로 보관한다. 변경 후 `python -m churn.api --export-openapi api/openapi.json`으로 갱신한다. 공개 예제는 합성 자료만 사용한다. `/docs`는 실행 중인 로컬 Swagger UI다.
