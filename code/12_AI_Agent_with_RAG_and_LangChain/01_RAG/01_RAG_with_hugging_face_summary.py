import wget
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch

import numpy as np
import random
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np

# You can also use this section to suppress warnings generated by your code:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# 파일 다운로드(1회용)
# filename = 'companyPolicies.txt'
# url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'
#
# # Use wget to download the file
# wget.download(url, out=filename)
# print('file downloaded')
##

"""
1. 문서 데이터 정리
회사 정책 텍스트 파일을 읽어서, 빈 줄을 제거하고 문단 단위 리스트로 나누는 함수
"""
def read_and_split_text(filename):
    # 파일을 UTF-8 인코딩으로 열어서 내용을 읽기
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()

    # 읽은 텍스트를 줄바꿈(\n) 기준으로 나누어 문단 리스트 생성
    paragraphs = text.split('\n')

    # 빈 줄이나 공백만 있는 항목 제거하고 양쪽 공백 제거
    paragraphs = [para.strip() for para in paragraphs if len(para.strip()) > 0]

    # 전처리된 문단 리스트 반환
    return paragraphs


# 텍스트 파일을 읽고 문단 단위로 분리하고 뒤섞음
paragraphs = read_and_split_text('../companyPolicies.txt')
random.shuffle(paragraphs)

# DPR 문맥 인코더용 토크나이저를 로드 (페이스북이 공개한 사전학습 모델)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# DPR 문맥 인코더 모델 불러오기
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

"""
2. 문서 데이터 임베딩 벡터로 변환
"""
def encode_contexts(text_list):
    # 입력된 텍스트 리스트를 임베딩 벡터로 변환하는 함수
    embeddings = []
    for text in text_list:
        # 각 문장(문단)을 토큰화
        inputs = context_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        # DPR 컨텍스트 인코더로 임베딩 생성
        outputs = context_encoder(**inputs)
        # 3D t-SNE 시각화
        # tsne_plot(outputs.pooler_output.detach().numpy())
        # 풀러 출력(pooler_output)을 리스트에 추가
        embeddings.append(outputs.pooler_output)
    # 모든 벡터를 하나의 큰 배열로 이어붙여서 반환
    return torch.cat(embeddings).detach().numpy()

# 위 함수를 사용해서 paragraphs 리스트를 임베딩 벡터로 변환
context_embeddings = encode_contexts(paragraphs)

# 임베딩 리스트를 단일 NumPy 배열로 변환
embedding_dim = 768  # 임베딩 차원 수 (모델 아키텍처에 맞춰야 함)
context_embeddings_np = np.array(context_embeddings).astype('float32')

import faiss

# FAISS의 L2 거리 기반 평면 인덱스 생성
index = faiss.IndexFlatL2(embedding_dim)
# 문맥 임베딩들을 인덱스에 추가
index.add(context_embeddings_np)

"""
3. 질문 인코더, 토크나이저 및 함ㅅ
"""
# DPR 질문 인코더와 토크나이저 로드
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

def search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5):
    """
    주어진 질문에 대해 가장 관련성이 높은 문맥들을 검색하는 함수.

    인자:
    - question: 질문 문장 (str)
    - question_tokenizer: 질문 인코더용 토크나이저
    - question_encoder: DPR 질문 인코더 모델
    - index: FAISS 인덱스
    - k: 상위 몇 개의 문맥을 반환할지

    반환값:
    - tuple: (거리 행렬, 인덱스 행렬)
    """

    # 질문을 토큰화
    question_inputs = question_tokenizer(question, return_tensors='pt')

    # 질문을 인코더를 통해 임베딩 벡터로 변환
    question_embedding = question_encoder(**question_inputs).pooler_output.detach().numpy()

    # FAISS 인덱스에서 질문 임베딩과 가장 유사한 k개의 문맥 검색
    D, I = index.search(question_embedding, k)

    return D, I

"""
4. 대답 인코더, 토크나이저 및 함수
"""
# GPT2 토크나이저와 모델 불러오기 (Hugging Face에서 제공하는 공개 커뮤니티 버전)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

# 생성 시 패딩 토큰 ID 설정 (에러 방지용)
model.generation_config.pad_token_id = tokenizer.pad_token_id

def generate_answer(question, contexts):
    # 검색된 컨텍스트들을 하나의 큰 문자열로 이어서 GPT2 입력으로 만들기
    input_text = question + ' ' + ' '.join(contexts)

    # 입력 텍스트를 토큰화 (패딩/자르기 포함)
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

    # GPT2를 사용해 답변 생성
    summary_ids = model.generate(
        inputs['input_ids'],
        max_new_tokens=50,  # 새로 생성할 최대 토큰 수
        min_length=40,  # 생성될 최소 길이
        length_penalty=2.0,  # 더 긴 답변을 선호하도록 페널티 조정
        num_beams=4,  # 빔 서치 사용해 더 좋은 결과 탐색
        early_stopping=True,  # 조건 만족하면 빔 서치 조기 종료
        pad_token_id=tokenizer.eos_token_id  # 패딩 토큰 설정
    )

    # 생성된 토큰 시퀀스를 사람이 읽을 수 있는 텍스트로 변환
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

"""
실제 실행
1. 질문 임베딩
2.faiss 검색 및 상위 문단 찾음
3. 질문과 상위 컨텍스트를 토큰나이징 및 답변 생성, 출력(디코드) 
"""
# 예제 질문
question = "what is Internet and Email Policy Acceptable Use?"
print("question: ", question)

# DPR 검색 함수로 가장 관련 있는 문단 상위 5개 찾기
_, I = search_relevant_contexts(question, question_tokenizer, question_encoder, index, k=5)

print(f"paragraphs indexs {I}")  # 검색된 문단들의 인덱스 출력

# 검색 결과 인덱스를 사용해 실제 문단 텍스트 가져오기
top_contexts = [paragraphs[idx] for idx in I[0]]
print(f"top_contexts {top_contexts}")  # 선택된 문단 내용 출력

# 질문과 상위 컨텍스트들을 GPT2에 넣어 최종 답변 생성
answer = generate_answer(question, top_contexts)
print("Generated Answer:", answer)  # 생성된 답변 출력