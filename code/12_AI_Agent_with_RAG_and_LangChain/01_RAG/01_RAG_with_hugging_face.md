RAG와 Hugging Face를 사용하여 LLM 향상시키기

시나리오
당신은 이제 HR 부서가 회사 정책에 대한 지능형 질의응답 도구를 구축하도록 돕기 위해 고용되었다고 상상해 보세요.
직원들은 “우리 휴가 정책이 어떻게 되나요?” 또는 “경비 청구는 어떻게 제출하나요?” 같은 질문을 입력할 수 있고, 이에 대해 즉각적이고 명확한 답변을 받을 수 있습니다.

이 도구는 방대한 정책 문서를 일일이 검색하지 않고도 관련 정보를 자동으로 제공함으로써 직원들의 시간을 절약하고 복잡한 정책을 쉽게 이해하도록 도와줄 것입니다.

개요
이번 실습에서는 Retriever-Augmented Generation (RAG) 라는 고급 개념을 다룹니다.
RAG는 **검색(Retrieval)**과 **생성(Generation)**의 장점을 결합한 최신 자연어 처리(NLP) 접근 방식입니다.

이 실습을 통해 다음을 배우게 됩니다:

대규모 데이터셋에서 관련 정보를 효과적으로 검색(Retrieve)
최신 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델을 활용해 정확한 답변 생성(Generate)
Dense Passage Retriever (DPR)과 GPT2 모델 같은 최첨단 도구를 통합함으로써, 질문에 대해 즉석에서 정보를 찾아 종합적인 답변을 생성하는 고급 QA 시스템을 구축하는 방법을 익히게 됩니다.

또한, 실습 중심의 코딩 과제를 통해 실제 NLP 과제를 다루는 경험을 쌓고, 견고한 NLP 파이프라인을 설계하며, 모델의 정확성과 적합성을 높이도록 파인튜닝하는 방법도 배우게 됩니다.

Objectives (목표)
이 실습을 마치면 다음을 할 수 있습니다:

RAG의 개념과 구성 요소 이해: Retrieval과 Generation 기술이 NLP에서 어떻게 결합되는지 이해
Dense Passage Retriever (DPR) 구현: 큰 데이터셋에서 문서를 효율적으로 검색하여 생성 모델에 공급하는 법 배우기
시퀀스-투-시퀀스 모델 통합: DPR이 제공한 컨텍스트를 바탕으로 GPT2 같은 모델로 답변을 생성해 정확성과 관련성 향상
질의응답 시스템 구축: DPR과 GPT2를 함께 활용한 QA 시스템을 실제로 개발
NLP 모델 파인튜닝 및 최적화: 특정 작업이나 데이터셋에 맞게 모델 성능을 개선
전문 NLP 도구 사용법 습득: Hugging Face의 transformers와 datasets 라이브러리 같은 고급 NLP 도구로 정교한 솔루션 구현

"""
✅ 1️⃣ max_length
설명: 생성할 텍스트의 최대 길이 (토큰 단위)

예상 변화:

값이 작으면 → 짧고 간결하지만 정보 부족할 수 있음

값이 크면 → 더 길게, 상세하게 설명하려 시도

주의: 너무 크면 → 중복, 횡설수설 가능성 증가

✅ 2️⃣ min_length
설명: 생성할 텍스트의 최소 길이

예상 변화:

값이 크면 → 더 긴 문장을 강제

답이 너무 짧게 끊기는 걸 방지

주의: 무조건 길어야 좋은 건 아님. 컨텍스트 없는 문장 추가될 수도 있음

✅ 3️⃣ length_penalty
설명: 긴 출력을 얼마나 선호/억제할지 제어

1.0 → 균형적

1.0 → 짧게 압축

<1.0 → 길게 풀어쓰기

예상 변화:

penalty > 1.0 → 더 간결하고 요약적인 답변

penalty < 1.0 → 풍부하고 장황해질 가능성

✅ 4️⃣ num_beams
설명: 빔서치 탐색 폭 (생성 다양성 ↔ 품질 트레이드오프)

예상 변화:

작은 값(1-2) → 빠르지만 간단하거나 뻔한 답

큰 값(4-8) → 더 정교하고 품질 좋은 문장, 문법적 일관성↑

주의: 너무 크면 계산 느려지고 창의성↓ (너무 평균적인 문장)
"""