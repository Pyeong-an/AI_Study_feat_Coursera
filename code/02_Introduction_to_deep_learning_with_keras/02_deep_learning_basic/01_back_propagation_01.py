import numpy as np  # 넘파이 라이브러리 임포트 (행렬/수치 계산을 위해)
import matplotlib.pyplot as plt  # 그래프를 그리기 위해 matplotlib 임포트

"""
이 함수는 랜덤으로 가중치와 바이어스를 정한 뒤,
X > d 정답에 맞춰가도록 테스트 한 것이다. 수많은 횟수를 통해 에러를 줄이는것을 볼 수 있다
"""

# 입력값과 정답을 정의 (XOR 진리표)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2x4 행렬, 각 열이 하나의 입력 샘플
d = np.array([0, 1, 1, 0])  # XOR 문제의 정답 출력

def initialize_network_parameters():
    # 네트워크 파라미터 설정
    inputSize = 2      # 입력 뉴런 수 (x1, x2)
    hiddenSize = 2     # 은닉층 뉴런 수
    outputSize = 1     # 출력 뉴런 수
    lr = 0.1           # 학습률
    epochs = 180000    # 학습 반복 횟수

    # 가중치와 바이어스를 [-1, 1] 범위로 랜덤 초기화
    w1 = np.random.rand(hiddenSize, inputSize) * 2 - 1  # 입력층 → 은닉층 가중치
    b1 = np.random.rand(hiddenSize, 1) * 2 - 1          # 은닉층 바이어스
    w2 = np.random.rand(outputSize, hiddenSize) * 2 - 1 # 은닉층 → 출력층 가중치
    b2 = np.random.rand(outputSize, 1) * 2 - 1          # 출력층 바이어스

    return w1, b1, w2, b2, lr, epochs  # 초기화된 파라미터 반환

# 초기화된 파라미터를 가져옴
w1, b1, w2, b2, lr, epochs = initialize_network_parameters()

# 오차 값을 저장할 리스트
error_list = []

# 학습 반복
for epoch in range(epochs):
    # ----- 순전파 -----
    z1 = np.dot(w1, X) + b1  # 은닉층 선형 결합 (가중합)
    a1 = 1 / (1 + np.exp(-z1))  # 은닉층 활성화 함수 (시그모이드)

    z2 = np.dot(w2, a1) + b2  # 출력층 선형 결합
    a2 = 1 / (1 + np.exp(-z2))  # 출력층 활성화 함수 (시그모이드)

    # ----- 오차 계산 및 역전파 -----
    error = d - a2  # 출력층 오차 (목표값 - 예측값)
    da2 = error * (a2 * (1 - a2))  # 출력층 활성화 함수의 도함수를 곱한 값
    dz2 = da2  # 출력층의 그래디언트

    # 은닉층으로 오차를 전파
    da1 = np.dot(w2.T, dz2)  # 은닉층으로의 오차 전파
    dz1 = da1 * (a1 * (1 - a1))  # 은닉층 활성화 함수의 도함수를 곱한 값

    # ----- 가중치와 바이어스 업데이트 -----
    w2 += lr * np.dot(dz2, a1.T)  # 은닉층 → 출력층 가중치 업데이트
    b2 += lr * np.sum(dz2, axis=1, keepdims=True)  # 출력층 바이어스 업데이트

    w1 += lr * np.dot(dz1, X.T)  # 입력층 → 은닉층 가중치 업데이트
    b1 += lr * np.sum(dz1, axis=1, keepdims=True)  # 은닉층 바이어스 업데이트

    # 10000번마다 평균 오차 출력 및 저장
    if (epoch+1)%10000 == 0:
        print("Epoch: %d, Average error: %0.05f"%(epoch, np.average(abs(error))))
        error_list.append(np.average(abs(error)))

# ----- 학습 후 테스트 -----
z1 = np.dot(w1, X) + b1  # 은닉층 선형 결합
a1 = 1 / (1 + np.exp(-z1))  # 은닉층 활성화 함수

z2 = np.dot(w2, a1) + b2  # 출력층 선형 결합
a2 = 1 / (1 + np.exp(-z2))  # 출력층 활성화 함수

# ----- 결과 출력 -----
print('Final output after training:', a2)  # 학습 후 최종 출력
print('Ground truth', d)  # 실제 정답
print('Error after training:', error)  # 마지막 오차
print('Average error: %0.05f'%np.average(abs(error)))  # 마지막 평균 오차

# ----- 오차 그래프 그리기 -----
plt.plot(error_list)
plt.title('Error')  # 제목
plt.xlabel('Epochs')  # x축 레이블
plt.ylabel('Error')  # y축 레이블
plt.show()
