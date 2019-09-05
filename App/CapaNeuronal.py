import numpy as np

class CapaNeuronal():
    def __init__(self,conn,neuron,act):
        self.weight=np.random.rand(conn,neuron) *2 -1
        self.bias=np.random.rand(1,neuron)* 2 - 1
        self.activacion=act
    def sigmoide(self,x):
        return 1 / (1 + np.exp(-x))
    def Dsigmoide(self,x):
        return self.sigmoide(x) * (1 - self.sigmoide(x))