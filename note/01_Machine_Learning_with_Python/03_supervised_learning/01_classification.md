💡 분류가 뭐야? 👉 분류는 무엇인지 맞히는 문제야!

그러니까,
사진을 보고 “강아지야? 고양이야?” 맞히는 것!
이메일이 “스팸이야? 아니야?” 구별하는 것!

✅ 분류는 어떻게 배우냐면?
예시를 많이 보여줘서 배우는 거야!

사진과 정답을 많이 보여주면 👉 컴퓨터가 “아! 이렇게 생기면 강아지구나!” 하고 배워.

✅ 분류가 쓰이는 곳
스팸 이메일 걸러내기
손글씨 숫자 알아보기
사진 보고 동물 구별하기
병원에서 약 선택 도와주기
은행에서 누가 돈 갚을지 예측하기
가게에서 고객이 뭘 좋아할지 예측하기

✅ 레이블이 뭐야?
정답!
예: “강아지”, “고양이”

컴퓨터가 배우는 건 “이 사진은 강아지다!” 같은 정답(레이블)을 맞히도록 하는 것이야.

✅ 분류 문제는 크게 두 가지!
1️⃣ 이진 분류 (Binary Classification)
👉 둘 중 하나 고르기
예: 스팸이야? 아니야?
예: 돈 갚을까? 안 갚을까?

2️⃣ 다중 클래스 분류 (Multi-Class Classification)
👉 셋 이상 중 하나 고르기
예: 약 A, 약 B, 약 C 중 어떤 약이 좋아?
예: 사진이 강아지, 고양이, 토끼 중 뭘까?

✅ 컴퓨터가 분류를 배우는 방법 (알고리즘)
컴퓨터가 배우는 방법에는 여러 가지가 있어:

로지스틱 회귀
나이브 베이즈
의사 결정 나무
K-최근접 이웃 (KNN)
서포트 벡터 머신 (SVM)
신경망

이건 모두 문제를 맞히는 방법을 가르치는 도구야!

✅ 클래스가 여러 개일 땐?
👉 컴퓨터는 보통 “이거야? 아니야?”만 잘해.

그래서 여러 가지 중에서 고를 땐

One-vs-All (원 대 전부) 방법
One-vs-One (원 대 원) 방법

📌 One-vs-All (원 대 전부)
“각 클래스 하나 vs 나머지 전부” 문제로 바꿔.

예: 사과, 바나나, 오렌지라면
사과냐 아니냐?
바나나냐 아니냐?
오렌지냐 아니냐?

이렇게 3개의 문제를 풀어보고 점수가 가장 높은 걸 정답으로 선택!

장점 - 학습 종류가 적음
단점 - 한 종류의 데이터가 적으면 학습이 어려움

📌 One-vs-One (원 대 원)
두 개씩 짝지어서 비교!

예: 사과, 바나나, 오렌지라면
사과 vs 바나나
사과 vs 오렌지
바나나 vs 오렌지

이렇게 모든 쌍을 만들어 문제를 풀어.

새 데이터를 넣으면 각 비교에서 누가 이겼는지 “투표” 제일 많이 이긴 게 정답!

장점 : 세밀하게 구별하기 좋음, 문제를 쪼개 나눌 수 있음
단점 : 너무 많으면 쌍 비교가 많아짐 n(n-1), 점수가 똑같을 경우 애매함

✅ 정리!

분류는 컴퓨터가 보고 “이게 뭐야?”를 맞히게 하는 것!
여러 방법을 배워서 잘 맞히게 하고,
여러 클래스가 있을 땐 원 대 전부, 원 대 원 전략으로 문제를 풀어!