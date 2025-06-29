import numpy as np # import Numpy library to generate

# 랜덤 값을 생성한다(weight :  가중치(계수), bias : 상수)( y = w x + b)
weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

print(weights)
print(biases)

# 인풋 값 설정
x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

# 계산식 설정(가중합)
z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

# 문제 :  Next, let's compute the weighted sum of the inputs, z 1, 2, at the second node of the hidden layer. Assign the value to z_12.
# z_12를 만들어보자
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]

print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))

# activation(첫번째 은닉 층의 출력값)
a_11 = 1.0 / (1.0 + np.exp(-z_11))

print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# 문제 : Let's also compute the activation of the second node, a 1, 2, in the hidden layer. Assign the value to a_12.
# a_12를 만들어보자
a_12 = 1.0 / (1.0 + np.exp(-z_12))

print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

# 문제 : Now these activations will serve as the inputs to the output layer. So, let's compute the weighted sum of these inputs to the node in the output layer. Assign the value to z_2.
# z_2를 만들어보자
z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]

print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

# 문제 : Finally, let's compute the output of the network as the activation of the node in the output layer. Assign the value to a_2.
# a_2를 만들어보자
a_2 = 1.0 / (1.0 + np.exp(-z_2))

print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))
