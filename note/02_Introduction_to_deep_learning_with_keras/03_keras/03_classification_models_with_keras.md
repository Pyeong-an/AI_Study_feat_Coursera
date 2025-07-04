🌟 💡 오늘 주제: Keras로 분류(Classification) 문제 풀기
✅ 1️⃣ 분류(Classification) 문제란?
질문: “분류가 뭐야?”

👉 여러 가지 종류 중 어느 하나인지 맞히는 문제야!

✅ 예시:

“이 차를 사는 게 좋은 선택일까? 나쁜 선택일까?”
→ "나쁨(0), 보통(1), 좋음(2), 아주 좋음(3)" 중 하나를 맞추는 거야!

✅ 2️⃣ 우리가 풀 문제 예시
차를 살지 말지 추천하는 문제야.

차에 대한 정보가 있어:

가격 (높음/중간/낮음)

유지비 (높음/중간/낮음)

몇 명 탈 수 있는지 (2명/2명 이상)

✅ 데이터는 표로 되어 있어.

각 줄이 차 한 대의 정보야.

마지막 열에는 “추천 정도”가 0~3 숫자로 있어.

✅ 예시 데이터:

가격	유지비	인원	결정
높음	높음	2명	0 (비추천)
낮음	낮음	4명	3 (아주 추천)

✅ 3️⃣ 데이터 나누기
딥러닝 모델이 배우기 쉽게 나눠야 해!
1️⃣ 입력값(Predictors): 가격, 유지비, 인원
2️⃣ 목표값(Target): 추천 정도 (0, 1, 2, 3)

✅ 👉 이렇게 나누면

“이런 차 정보 → 추천 정도” 를 배우게 할 수 있어!

✅ 4️⃣ 분류 문제에서 주의할 점
목표값(타겟)이 그냥 숫자(0,1,2,3)라고 그대로 쓰면 안 돼!

컴퓨터가 헷갈려해.

“0이 1보다 작네, 2가 3보다 크네” 이런 숫자 계산으로 오해할 수 있어.

✅ 👉 그래서 어떻게 해?

각 숫자를 특별한 코드로 바꿔야 해!

이걸 **원-핫 인코딩(One-Hot Encoding)**이라고 불러.

✅ 예시:

0 → [1,0,0,0]

1 → [0,1,0,0]

2 → [0,0,1,0]

3 → [0,0,0,1]

✅ 이렇게 바꿔야 컴퓨터가 “이건 0번 클래스야!”라고 정확히 알 수 있어.

✅ 5️⃣ Keras의 to_categorical 함수
이런 변환을 쉽게 해주는 마법 같은 함수!

한 줄 코드로 목표값을 위처럼 바꿀 수 있어.

✅ 👉 장점:

실수 없이 쉽게 바꿀 수 있어.

코드가 짧고 간단해.

✅ 6️⃣ 신경망(Neural Network) 구조 만들기
Keras를 써서 쉽게 신경망을 만들어.

✅ 층(layer) 구성:

입력층: 8개 정보 넣기

첫 번째 숨겨진 층: 5개 노드

두 번째 숨겨진 층: 5개 노드

출력층: 4개 노드

왜 4개? → 4개의 추천 클래스(0,1,2,3)가 있으니까!

✅ 활성화 함수:

숨겨진 층 → ReLU

장점: 계산 빠르고 잘 배우게 해 줌.

출력층 → Softmax

장점: 결과를 확률로 만들어서 4개 값이 합쳐서 1이 되도록 해 줌.

각 값이 “이 클래스일 확률”이 돼!

✅ 7️⃣ 손실 함수(Loss Function)
모델이 틀린 만큼 얼마나 틀렸는지 계산.

분류 문제에서는 → Categorical Crossentropy(범주형 교차 엔트로피) 사용.

✅ 👉 왜 이걸 써?

여러 가지 분류 중 정답을 잘 맞추도록 배우게 해 줌.

Softmax 출력과 잘 어울려.

✅ 8️⃣ 평가 지표(Metric)
얼마나 잘 맞추는지 측정하는 방법.

여기선 → Accuracy(정확도) 사용.

예측이 맞았는지 안 맞았는지 비율!

✅ 👉 장점:

직관적이고 이해하기 쉬워.

많이 맞을수록 숫자가 높아져서 좋아.

✅ 9️⃣ 모델 학습하기
fit() 메서드로 학습.

입력값과 목표값 넣고

에포크(epochs) 횟수 지정.

✅ 에포크란?

데이터 전체를 한 번 학습하는 것.

여러 번 반복해서 점점 더 잘 배우게 함.

✅ 10️⃣ 예측하기
predict() 메서드 사용.

출력값 → 각 클래스일 확률!

✅ 예시:

[0.99, 0.01, 0.00, 0.00]
→ 거의 확실하게 0번 클래스!

✅ 모델 해석:

확률이 가장 높은 클래스가 예측 결과야.

확률이 비슷하면 → 모델이 덜 확실함.

✅ 11️⃣ 핵심 요약
✅ Keras로 분류 문제 풀기
1️⃣ 데이터를 입력값과 목표값으로 나누기
2️⃣ 목표값을 원-핫 인코딩으로 바꾸기 (to_categorical)
3️⃣ 모델 만들기

숨겨진 층 ReLU

출력층 Softmax
4️⃣ 손실 함수: Categorical Crossentropy
5️⃣ 평가 지표: Accuracy
6️⃣ fit()으로 학습
7️⃣ predict()로 확률 예측

✅ 12️⃣ 아주 쉽게 한 문장으로!
“Keras는 차 정보를 보고 ‘이 차를 사면 좋을까 나쁠까’를 똑똑하게 예측해 주는 컴퓨터 뇌를 쉽게 만들 수 있는 도구 상자야!”