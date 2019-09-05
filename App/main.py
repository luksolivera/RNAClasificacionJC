from App.DataSet import DataSet as data
import App.RedNeuronal as Red
import matplotlib.pyplot as plt
import numpy as np

n=200
f=0.4
desorden = 0.1
X,Y,p=data.getDataset(n,f,desorden)

Y=Y[:,np.newaxis]
#data.show(X,Y)                 #Para graficar los datos de entrada.

#Variables de la red.
arq= [p,8,1]                    #capas
epocas=2000                    #cant iteraciones
alfa= 0.01                     #aprendizaje
act='sigmoide'                  #funcion de activacion
loss=[]                         #Array de error de entrenamiento.


neural_n = Red.create_net(arq,act)

# COMIENZA LA ITERACIÓN

for i in range (epocas+1):
    # Entrenamiento de la red.
    result= Red.train(neural_n,X,Y,alfa)
    if i % 50 == 0:
        loss.append(Red.cost(result,Y))

        res=50
        """Array random para la predicción"""
        _x0 = np.linspace(-1.5,1.5,res)
        _x1 = np.linspace(-1.5,1.5,res)

        _Y= np.zeros((res,res))

        # Predición
        for i0, x0 in enumerate(_x0):
            for i1,x1 in enumerate(_x1):
                _Y[i0,i1] = Red.train(neural_n,np.array([[x0,x1]]),Y,alfa,train=False)

        # Gráfico
        plt.ion()
        fig2 = plt.figure("Función de Error")
        fig2.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.plot(range(len(loss)), loss)
        fig1 = plt.figure("Predicción")
        fig1.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        plt.axis("equal")
        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

        if i == epocas:
            print('fin de la ejecución')
            plt.ioff()
            plt.show()

        else:
            plt.draw()
            plt.pause(0.01)





