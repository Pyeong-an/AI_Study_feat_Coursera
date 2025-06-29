import numpy as np  # Numpy 라이브러리를 임포트 (배열, 랜덤값 생성 등 수학 계산에 사용)
from random import seed

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs  # number of nodes in the previous layer

    network = {}

    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):

        if layer == num_hidden_layers:
            layer_name = 'output'  # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1)  # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer]

            # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node + 1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network  # return the network

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


def forward_propagate(network, inputs):
    layer_inputs = list(inputs)  # start with the input layer as the input to the first hidden layer

    for layer in network:

        layer_data = network[layer]

        layer_outputs = []
        for layer_node in layer_data:
            node_data = layer_data[layer_node]

            # compute the weighted sum and the output of each node at the same time
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))

        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))

        layer_inputs = layer_outputs  # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions

small_network = initialize_network(5, 3, [3, 2, 3], 1)

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

# 문제 : Use the compute_weighted_sum function to compute the weighted sum at the first node in the first hidden layer.
# 첫번째 은닉층의 첫번째 노드의 계산을 보여라
weighted_sum = compute_weighted_sum(inputs, small_network['layer_1']['node_1']['weights'], small_network['layer_1']['node_1']['bias'])
print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))

# 문제 : Use the node_activation function to compute the output of the first node in the first hidden layer.
# 계산한 노드의 출력값은?
node_output = node_activation(weighted_sum)
print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))

# 문제 : Use the forward_propagate function to compute the prediction of our small network
# 포워드 프로퍼게이트를 사용해서 이 네트워크의 출력값을 구하라
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))