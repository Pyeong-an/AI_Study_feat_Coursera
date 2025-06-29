import numpy as np  # 넘파이 라이브러리 불러오기 (수학 계산용)
import matplotlib.pyplot as plt  # 맷플롯립의 pyplot 모듈 불러오기 (그래프 그리기용)

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  # 시그모이드 수식: 1 / (1 + e^-z)

# 시그모이드 함수의 도함수(미분값) 정의
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))  # 시그모이드의 미분 공식: σ(z) * (1 - σ(z))

def tanh(z):
    return np.tanh(z)  # 넘파이 내장 함수 사용 (e^z - e^-z) / (e^z + e^-z)

# 하이퍼볼릭 탄젠트 함수의 도함수(미분값) 정의
def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2  # tanh의 미분 공식: 1 - tanh(z)^2

# ReLU 함수 정의
def relu(z):
    return np.maximum(0, z)  # ReLU 수식: max(0, z)

# ReLU 함수의 미분
def relu_derivative(z):
    return np.where(z > 0, 1, 0)  # z가 0보다 크면 1, 아니면 0

# Generate a range of input values
z = np.linspace(-5, 5, 100)

tanh_grad = tanh_derivative(z)
relu_grad = relu_derivative(z)

# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, tanh(z), label='Tanh Activation', color='b')
plt.plot(z, tanh_grad, label="Tanh Derivative", color='r', linestyle='--')
plt.title('Tanh Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()