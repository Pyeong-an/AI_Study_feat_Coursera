# ONEDNN 최적화와 로그 레벨 설정 (경고 제거 및 환경 설정)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 필요한 라이브러리 임포트
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from keras.layers import Layer
from keras.optimizers import Adagrad
import warnings
warnings.simplefilter('ignore', FutureWarning)  # FutureWarning 무시

############### 입/출력 데이터
# 샘플 문장 데이터 (영어->스페인어)
input_texts = [
    "Hello.", "How are you?", "I am learning machine translation.", "What is your name?", "I love programming."
]
target_texts = [
    "Hola.", "¿Cómo estás?", "Estoy aprendiendo traducción automática.", "¿Cuál es tu nombre?", "Me encanta programar."
]

# 디코더 입력에 시작과 끝 토큰 추가
target_texts = ["startseq " + x + " endseq" for x in target_texts]
############################

# 입력 언어 토크나이저 생성 및 시퀀스로 변환
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)

# 출력 언어 토크나이저 생성 및 시퀀스로 변환
output_tokenizer = Tokenizer()
output_tokenizer.fit_on_texts(target_texts)
output_sequences = output_tokenizer.texts_to_sequences(target_texts)

# 어휘 크기 계산
input_vocab_size = len(input_tokenizer.word_index) + 1
output_vocab_size = len(output_tokenizer.word_index) + 1

# 최대 시퀀스 길이 계산 및 패딩
max_input_length = max([len(seq) for seq in input_sequences])
max_output_length = max([len(seq) for seq in output_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_input_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_output_length, padding='post')

# 디코더 입력과 출력 데이터 준비
decoder_input_data = output_sequences[:, :-1]
decoder_output_data = output_sequences[:, 1:]

# 디코더 출력 데이터를 원-핫 인코딩
decoder_output_data = np.array([np.eye(output_vocab_size)[seq] for seq in decoder_output_data])

# 커스텀 Self-Attention 레이어 정의
class SelfAttention(Layer):
    def __init__(self, initializer='glorot_uniform', **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        # Q, K, V를 위한 가중치 행렬 생성(Query, Key, Value)
        self.Wq = self.add_weight(shape=(feature_dim, feature_dim),
                                  initializer=self.initializer,
                                  trainable=True,
                                  name='Wq')
        self.Wk = self.add_weight(shape=(feature_dim, feature_dim),
                                  initializer=self.initializer,
                                  trainable=True,
                                  name='Wk')
        self.Wv = self.add_weight(shape=(feature_dim, feature_dim),
                                  initializer=self.initializer,
                                  trainable=True,
                                  name='Wv')
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        # 입력으로부터 Query, Key, Value 계산
        q = K.dot(inputs, self.Wq)
        k = K.dot(inputs, self.Wk)
        v = K.dot(inputs, self.Wv)
        # 스케일된 닷 프로덕트 어텐션 스코어 계산
        scores = K.batch_dot(q, k, axes=[2, 2])
        scores = scores / K.sqrt(K.cast(K.shape(k)[-1], dtype=K.floatx()))
        attention_weights = K.softmax(scores, axis=-1)
        # 가중합 출력 생성
        output = K.batch_dot(attention_weights, v)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

def build_model(SelfAttentionClass, initializer='glorot_uniform', optimizer='adam'):
    # 인코더 정의
    encoder_inputs = Input(shape=(max_input_length,))  # 인코더 입력
    encoder_embedding = Embedding(input_vocab_size, 256)(encoder_inputs)  # 임베딩
    encoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)  # LSTM 출력과 상태
    encoder_states = [state_h, state_c]  # 인코더 상태 저장

    # 인코더에 Self-Attention 적용
    attention_layer = SelfAttentionClass(initializer=initializer)(encoder_outputs)

    # 디코더 정의
    decoder_inputs = Input(shape=(max_output_length - 1,))  # 디코더 입력
    decoder_embedding = Embedding(output_vocab_size, 256)(decoder_inputs)  # 임베딩
    decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)  # 초기 상태를 인코더 상태로
    decoder_attention = SelfAttentionClass(initializer=initializer)(decoder_outputs)  # 디코더에 Self-Attention 적용
    decoder_dense = Dense(output_vocab_size, activation='softmax')  # 출력층
    decoder_outputs = decoder_dense(decoder_attention)  # 최종 출력

    # 모델 생성
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # 컴파일

    # 모델 요약 출력
    # model.summary()
    return model

model_glorot = build_model(SelfAttention, 'glorot_uniform', 'adam')
model_he_adam = build_model(SelfAttention, 'he_uniform', 'adam')
model_he_adagrad = build_model(SelfAttention, 'he_uniform', 'adagrad')

# 모델 학습
history_glorot_adam = model_glorot.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)
history_he_adam = model_he_adam.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)
history_he_adagrad = model_he_adagrad.fit([input_sequences, decoder_input_data], decoder_output_data, epochs=100, batch_size=16)

# 학습 손실 비교 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history_glorot_adam.history['loss'], label='glorot + adam')
plt.plot(history_he_adam.history['loss'], label='He + adam')
plt.plot(history_he_adagrad.history['loss'], label='He + adagrad')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()