import math
class Network:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        layer_input = inputs
        for w_matrix, b_vector in zip(self.weights, self.biases):
            layer_output = []
            for j in range(len(w_matrix[0])): 
                z = 0
                for k in range(len(layer_input)): 
                    z += layer_input[k] * w_matrix[k][j]
                z += b_vector[j] 
                layer_output.append(z)
            layer_input = layer_output
        return layer_input
    
#Network1
weights_n1 = [
    [[0.5, 0.6],   # x1 → h1,h2
     [0.2, -0.6]], # x2 → h1,h2
    [[0.8],        # h1 → o1
     [0.4]]        # h2 → o1
]

biases_n1 = [
    [0.3, 0.25],   # hidden 層 bias
    [-0.5]         # output 層 bias
]

nn1 = Network(weights_n1, biases_n1)

inputs_set1 = [
    [1.5, 0.5],
    [0, 1]
]

print("=== Network1 Outputs ===")
for inp in inputs_set1:
    output = nn1.forward(inp)
    print(f"Input {inp} → Output {output}")

#Network2
weights2 = [
    # 第一層: input → h1,h2
    [[0.5, 0.6],   # x1
     [1.5, -0.8]], # x2
    # 第二層: h1,h2 → i1
    [[0.6],        # h1 → i1
     [-0.8]],      # h2 → i1
    # 第三層: i1 → o1,o2
    [[0.5, -0.4]]  # i1 → o1,o2
]

biases2 = [
    [0.3, 1.25],  # hidden 層1 bias: h1,h2
    [0.3],        # hidden 層2 bias: i1
    [0.2, 0.5]    # output 層 bias: o1,o2
]

nn = Network(weights2, biases2)

inputs_set2 = [
    [0.75, 1.25],
    [-1, 0.5]
]

print("=== Network2 Outputs ===")
for inp in inputs_set2:
    output = nn.forward(inp)
    print(f"Input {inp} → Output {output}")