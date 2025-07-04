PyTorch로 구현하는 RAG

소개 (Introduction)
당신은 한 소셜 미디어 회사에 고용된 머신러닝 엔지니어라고 상상해 보세요.
당신의 과제는 플랫폼에서 공유되는 노래가 아동에게 적합한지 여부를 판단하는 것입니다.

그러나 각 노래를 대형 언어 모델(LLM)을 사용해 직접 평가하는 것은 비용이 매우 높습니다.
이에 대한 대안으로 Retriever-Augmented Generation(RAG) 방식을 사용하기로 제안합니다.

RAG는 검색기(retriever) 모델과 생성기(generator) 모델의 장점을 결합합니다.

검색기 모델은 이미 답변된 콘텐츠 적합성 질문의 임베딩에서 관련 정보를 검색합니다.

생성기 모델은 이 정보를 활용해 새로운 콘텐츠(노래)의 적합성을 예측합니다.

이 접근법은 **평가 프로세스를 효율적으로 확장(scale)**하면서도,
각 노래의 콘텐츠가 아동 안전 측면에서 충분히 검토되도록 보장합니다.
즉, 모든 노래를 매번 대형 LLM에 넣어 돌리는 비용을 줄이면서도 품질 높은 심사가 가능합니다.

Objectives (목표)

이 실습을 마치면 다음을 할 수 있습니다:

임베딩 기술 이해: NLP 작업을 위해 사전 학습된(pre-trained) 모델로부터 임베딩을 생성하고 활용하는 법 학습

Hugging Face Transformers와 PyTorch 사용: transformers 라이브러리를 사용해 BERT 같은 고급 NLP 모델을 PyTorch에서 불러오고 활용

t-SNE를 활용한 시각화: 고차원 데이터를 저차원 공간에서 시각화(t-SNE)하여 데이터 분포와 클러스터를 이해

언어 모델 파인튜닝: 특정 작업을 위해 사전 학습 언어 모델을 파인튜닝하여 모델 성능 향상

실전 NLP 솔루션 개발: Retriever와 Generator 아키텍처를 효과적으로 사용해 질문에 답하는 시스템을 구현, 모델 튜닝부터 배포까지 엔드투엔드 워크플로우 경험

코사인 유사도 구현: 기존의 dot product 유사도 대신 cosine similarity를 사용하여 응답 생성 시스템의 관련성 검출 개선

모델 성능 평가: 질의응답(QA) 시스템에서 dot product 대신 cosine similarity 사용 시 검색 정확도와 관련성에 미치는 영향 평가








