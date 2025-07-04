✅ 먼저! 회귀(Regression)가 뭐야?
👉 숫자를 예측하는 문제야!
예:

시험 점수 맞히기

키와 나이를 보고 몸무게 예측하기

✅ 왜 평가가 필요해?
모델이 얼마나 잘 맞히는지 알아야 하니까!

예측이 진짜랑 비슷해야 쓸모가 있어.

✅ 오차(Error)가 뭐야?
👉 틀린 정도야.

실제 값과 예측 값의 차이!

예: 시험 점수 예측이 85점인데 실제는 80점 → 오차는 5점

✅ 평가 지표 4가지 배우자!
🌟 1️⃣ MAE (Mean Absolute Error)
✅ 뜻

예측이 얼마나 틀렸는지 평균적으로 알려줘.

“틀린 양”을 그냥 더해서 평균 냄.

✅ 계산

모든 오차의 절댓값을 더하고

개수로 나눔.

✅ 예시

예측 오차가 5점, 7점, 3점이면
→ MAE = (5 + 7 + 3) / 3 = 5

✅ 장점
✔️ 해석이 쉬워! → “평균적으로 몇 점 틀렸어”
✔️ 큰 오차가 너무 심하게 영향 안 줌

✅ 단점
✘ 너무 큰 오차(이상치) 반영이 약해 → 큰 실수에 덜 민감

🌟 2️⃣ MSE (Mean Squared Error)
✅ 뜻

오차를 제곱해서 평균

큰 오차를 더 크게 벌점 줌!

✅ 계산

오차를 각각 제곱 → 다 더함 → 나눔

✅ 예시

오차 5, 7, 3이면
→ 5² + 7² + 3² = 25 + 49 + 9 = 83
→ MSE = 83 / 3 ≈ 27.7

✅ 장점
✔️ 큰 오차를 강하게 벌줌 → 이상치 민감
✔️ 수학적으로 다루기 좋음

✅ 단점
✘ 단위가 제곱이라 직관적이지 않아
→ “점수²” 같은 단위라 해석이 어려움

🌟 3️⃣ RMSE (Root Mean Squared Error)
✅ 뜻

MSE의 제곱근

MSE 단위를 원래 단위로 돌려줌

✅ 계산

RMSE = √MSE

✅ 예시

위에서 MSE ≈ 27.7
→ RMSE ≈ √27.7 ≈ 5.26

✅ 장점
✔️ 단위가 예측 값이랑 같아 → “점수 차이”로 바로 이해 가능
✔️ 큰 오차에 민감 → 큰 실수 잘 잡아냄

✅ 단점
✘ 이상치가 있으면 너무 커질 수 있어 → 민감

🌟 4️⃣ R² (R-squared, 결정계수)
✅ 뜻

모델이 얼마나 잘 설명하는지 비율로 알려줘

0에서 1 사이 → 1이면 완벽!

1 - (오차(모델) / 오차(평균값만 쓴 경우))
유일하게 클수록 좋음

✅ 해석

R² = 0 → 못 맞힘

R² = 1 → 완벽하게 맞힘

0.85라면 → 85%를 설명함

✅ 예시

친구 시험점수 예측에서 R² = 0.85
→ “친구 점수 차이를 85%나 잘 맞혔어!”

✅ 장점
✔️ 직관적 → 비율로 설명
✔️ 비교하기 쉬움
✔️ 비전문가도 이해 가능

✅ 단점
✘ 선형 관계만 가정 → 곡선 모양 예측에는 부정확
✘ 이상치에 민감
✘ 음수가 나올 수도 → 아주 못 맞힌 모델

✅ 표로 정리
지표	뜻	장점	단점
MAE	평균 절댓값 오차	해석 쉽다	큰 오차 영향 적음
MSE	평균 제곱 오차	큰 오차 잘 잡음	단위가 제곱 → 직관 어려움
RMSE	MSE의 제곱근	해석 쉽다, 큰 오차 민감	이상치에 너무 민감
R²	설명된 비율	직관적, 비교 좋음	선형만 가정, 이상치 민감

✅ 추가 팁!
✨ 실제로는 한 가지 지표만 쓰지 않아
→ 여러 지표를 함께 보고 판단!

✅ 예:

MAE → 전반적으로 얼마나 틀렸나

RMSE → 큰 실수 있는지 확인

R² → 얼마나 잘 설명했나

✅ 아주 간단 한 문장 요약!

“회귀 모델 평가는 얼마나 잘 맞췄는지, 얼마나 많이 틀렸는지 숫자로 보여주는 것!”