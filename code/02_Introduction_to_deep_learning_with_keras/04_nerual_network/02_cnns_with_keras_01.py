# ONEDNN 최적화와 로그 레벨 설정 (경고 최소화)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 일부 CPU 최적화 끄기 (호환성 문제 예방용)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 경고 메시지를 최소화 (INFO/WARNING 숨김)

# Keras 라이브러리 불러오기
import keras
from keras.models import Sequential          # 순차 모델 생성
from keras.layers import Dense               # 완전연결(Dense) 계층
from keras.layers import Input               # 입력 계층
from keras.utils import to_categorical       # 라벨을 원-핫 인코딩

from keras.layers import Conv2D              # 합성곱 계층
from keras.layers import MaxPooling2D        # 풀링 계층
from keras.layers import Flatten             # 평탄화 계층

# MNIST 데이터셋 불러오기
from keras.datasets import mnist

# MNIST 데이터셋 로드 (훈련/테스트)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 이미지 데이터 형태 변환: (샘플 수, 28, 28, 1) 채널 차원 추가
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# 픽셀값 정규화 (0~1 범위로 변환)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 라벨을 원-핫 인코딩으로 변환
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 분류 클래스 개수 확인
num_classes = y_test.shape[1]

# CNN 모델 정의 함수
def convolutional_model():
    model = Sequential()  # 순차 모델 생성
    model.add(Input(shape=(28, 28, 1)))                               # 입력 계층
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu'))  # 합성곱 계층 (16필터, 5x5커널)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))         # 최대 풀링 계층 (2x2)

    model.add(Flatten())                          # 1차원 벡터로 평탄화
    model.add(Dense(100, activation='relu'))      # 은닉층 (뉴런 100개)
    model.add(Dense(num_classes, activation='softmax'))  # 출력층 (클래스 수만큼 뉴런, 소프트맥스)

    # 모델 컴파일 (옵티마이저, 손실함수, 평가메트릭 설정)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 모델 생성
model = convolutional_model()

# 모델 학습
model.fit(
    X_train, y_train,                     # 입력 데이터와 정답
    validation_data=(X_test, y_test),    # 검증 데이터
    epochs=10,                           # 학습할 에폭 수
    batch_size=200,                      # 배치 크기
    verbose=2                            # 출력 로그 레벨
)

# 테스트 데이터로 평가
scores = model.evaluate(X_test, y_test, verbose=0)

# 정확도와 오류율 출력
print("Accuracy: {} \n Error: {}".format(scores[1], 100 - scores[1]*100))
