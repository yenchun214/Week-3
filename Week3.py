import math
class Network:
    def __init__(self,weights):
        self.weights=weights
    
    def sigmoid(self,x):
        return 1/(1+math.exp(-x))
        
    def forward(self,inputs):
        layer_input=inputs
        for w_metrix in self.weights:
            layer_output=[]
            for j in range(len(w_metrix[0])):
                sum_neuro=0
                for k in range(len(layer_input)):
                    sum_neuro+=layer_input[k]*w_metrix[k][j]
                a=self.sigmoid(sum_neuro)
                layer_output.append(a)
            layer_input=layer_output
        return layer_input
    
#Network1
w_input_hidden=[
    [0.5,0.6],
    [0.2,-0.6],
    [0.3,0.25]
    ]
w_hidden_output=[
    [0.8],
    [0.4],
    [-0.5]
    ]
input_num=[
    [1.5,0.5,1],
    [0,1,1]
    ]
weight_n=[w_input_hidden,w_hidden_output]
nn=Network(weight_n)
for inp in input_num:
    outputs=nn.forward(inp)
    print(f"input:{inp},outputs:{outputs}")

#Network2
w_input_hidden2=[
    [0.5,0.6],
    [1.5,-0.8],
    [0.3,1.25]
]
w_hidden_hidden2=[
    [0.6],
    [-0.8],
    [0.3]
]
w_hidden_output2=[
    [0.5,-0.4],
    [0.2,0.5]
]
input_num2=[
    [0.75,1.25,1],
    [-1,0.5,1]
]

weight_n2=[w_input_hidden2,w_hidden_hidden2,w_hidden_output2]
nn=Network(weight_n2)
for inp in input_num2:
    outputs=nn.forward(inp)
    print(f"input:{inp},outputs:{outputs}")