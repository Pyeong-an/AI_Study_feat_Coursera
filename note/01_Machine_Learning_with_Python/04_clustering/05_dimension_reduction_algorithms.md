✅ 먼저! 차원축소가 뭐야?
👉 너무 많은 정보를 중요한 정보만 남기고 줄이는 것이야!

✅ 예시

100색 색연필을 → 빨강, 노랑, 파랑 3가지로 요약

얼굴 사진에서 → 진짜 얼굴 구별에 중요한 특징만 뽑기

✅ 왜 해?

계산을 빠르게 하고

보기 쉽게 만들고

중요한 정보만 쓰도록!

✅ 차원축소 알고리즘 3가지 배우자!
1️⃣ PCA (주성분 분석)
✨ 뭐 하는 거야?

데이터를 **가장 중요한 방향(축)**으로 줄여서 정리

비슷한 정보는 합치고 → 덜 중요한 건 버림

✅ 비유로!

큰 색연필 세트에서 거의 같은 색끼리 → 대표 색 하나로!

너무 비슷한 건 하나만 남기면 되니까

✅ 장점

빠르고 간단해!

선처럼 곧은(직선적인) 패턴을 잘 찾음

노이즈(쓸모없는 흔들림)도 줄여줘

✅ 단점

직선적인(선형) 관계만 잘 잡음

너무 꼬불꼬불한 모양은 못 잡아

✅ 어디에 좋아?

데이터가 곧게 퍼져 있거나 규칙적인 경우

얼굴 사진 → 중요한 얼굴 모양 요약

2️⃣ t-SNE
✨ 뭐 하는 거야?

데이터가 고차원이라 복잡할 때 → 2D나 3D로 예쁘게 펼쳐서 보기 좋게!

비슷한 것끼리 가까이, 다른 것끼리 멀리 배치

✅ 비유로!

반 친구들을 키, 몸무게, 취미로 분류 → 2D 종이에 “친한 애들끼리 가까이” 그려주는 것

✅ 장점

비슷한 그룹(클러스터)을 아주 잘 보여줘!

복잡한 데이터도 보기 좋게

✅ 단점

느려

설정(하이퍼파라미터)이 까다로움

새로운 데이터 넣으려면 처음부터 다시 계산해야 함

✅ 어디에 좋아?

그림으로 클러스터를 예쁘게 보고 싶을 때

복잡한 데이터 → 그룹을 발견하고 싶을 때

이미지나 글자 분류할 때

3️⃣ UMAP
✨ 뭐 하는 거야?

t-SNE랑 비슷하지만 더 똑똑하고 빠르게!

데이터의 **모양(매니폴드)**을 잘 유지하면서 낮은 차원으로 줄여줌

전체적인 구조도 잘 보존

✅ 비유로!

친구들이 반에 앉아있는 모양을 → 종이에 옮기되

친한 애들끼리는 가깝게

반 전체 자리 배치 모양도 유지!

✅ 장점

빠름

큰 데이터도 잘 됨

전체적인 구조 + 세부적인 클러스터 → 둘 다 잘 유지

새 데이터도 쉽게 넣을 수 있음

✅ 단점

아주 복잡할 땐 잘못 나누기도 해

설정도 좀 조절해봐야 해

✅ 어디에 좋아?

데이터가 복잡하고 큰 경우

클러스터 보고 싶고

나중에도 새로운 데이터 추가하고 싶을 때

✅ 세 알고리즘 비교 요약
알고리즘	잘하는 것	장점	단점
PCA	직선적인 정보 요약	빠르고 간단	비선형 모양은 못 잡음
t-SNE	비슷한 것끼리 묶기 시각화	클러스터를 예쁘게 그림	느리고 새 데이터 추가 어려움
UMAP	구조 유지하며 차원 축소	빠르고 큰 데이터에 좋아	설정이 조금 까다로움

✅ 한 문장 요약!

PCA → “간단하고 빠르게 중요한 정보 줄이기”
t-SNE → “비슷한 것끼리 잘 묶어서 보기 좋게”
UMAP → “빠르고 구조도 잘 유지하면서 똑똑하게 줄이기”