import numpy as np

from App.CapaNeuronal import CapaNeuronal

"""Funciones de activacion y sus derivadas"""
sigm = lambda x: 1 / (1 + np.e ** (-x))
dSigm = lambda x: x * (1 - x)
cost = lambda Yp,Yr: np.mean((Yp - Yr) ** 2)
dCost= lambda Yp,Yr: (Yp - Yr)

def create_net(arq, act):
    # arq :  array con la arquitectura de la red
    # act: funcion de activacion
    net = []
    for l, layer in enumerate(arq[:-1]):
        net.append(CapaNeuronal(arq[l], arq[l+1], act))

    return net

"""net: red neuronal
    X: datos de entrada, Y datos salida
    fCost: funcion de coste
    alfa: aprendizaje"""
def train(net, X , Y,alfa,train=True):
    #FORDWARD
    out = predict(net, X)

    #BACKPROPAGATION
    deltas = []
    if train:
        for l in reversed(range(0, len(net))):
            salida = out[l+1][0]
            if l == (len(net)-1):
                # capa de salida
                cal = dCost(salida, Y) * eval('net[l].D' + net[l].activacion + '(salida)')
                deltas.insert(0, cal)
            else:# capa oculta
                cal = deltas[0].dot(_weight.T) * eval('net[l].D' + net[l].activacion + '(salida)')
                deltas.insert(0, cal)

            _weight = net[l].weight
        # GRADIENT DESEND
            net[l].bias = net[l].bias - np.mean(deltas[0], axis=0, keepdims=True) * alfa
            net[l].weight = net[l].weight - out[l][1].T.dot(deltas[0]) * alfa

    return out[-1][1]

"""net: red neuronal
   X: datos de entrada """
def predict(net,X):
    out = [(None, X)] # entradas, salidas
    for l, layer in enumerate(net):
        entrada = out[-1][1].dot(layer.weight) + layer.bias
        act = layer.activacion
        salida = eval('layer.' + act + '(entrada)')
        out.append((entrada, salida))

    return out

