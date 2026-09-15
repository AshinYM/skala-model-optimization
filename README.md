# 고객 이탈 예측 모델 개발 및 최적화

Bank Churn 데이터를 기반으로 비즈니스 관점의 목표를 먼저 정의하고, 전처리·Feature Engineering·Random Forest·하이퍼파라미터 비교를 수행한 프로젝트입니다.

## 설계 기준

- 이탈 고객을 놓치지 않도록 Recall과 ROC-AUC를 주요 기준으로 사용했습니다.
- 날짜 컬럼에서 계정 유지 기간을 파생하고, 결측값 대체와 범주형 One-Hot 인코딩을 Pipeline으로 묶었습니다.
- Validation 성능으로 모델을 선택한 뒤 Train+Validation으로 재학습하고 Test는 최종 1회 평가했습니다.

## 실행

```bash
jupyter lab
```

`종합실습_모델개발및최적화_신용민.ipynb`를 실행하면 실험 결과와 최종 예측 파일을 재현할 수 있습니다.
