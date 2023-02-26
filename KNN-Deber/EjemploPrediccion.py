#Tenemos un conjunto de datos con dos características
#(la longitud y el Ancho de los pétalos de una flor)
#y queremos predecir la clase de una flor desconocida.
# Para hacer esto, utilizaremos K-NN para encontrar los k vecinos más cercanos
#a la flor desconocida y predecir su clase en función de la clase mayoritaria de sus vecinos más cercanos

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Generamos un conjunto de datos de ejemplo
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
y = np.array([0, 0, 0, 1, 1, 1])

# Creamos una instancia del clasificador K-NN con k = 3
knn = KNeighborsClassifier(n_neighbors=3)

# Ajustamos el modelo a los datos
knn.fit(X, y)

# Flor desconocida
unknown_flower = np.array([[3, 2]])

# Predecimos la clase de la flor desconocida
predicted_class = knn.predict(unknown_flower)

print("La clase de la flor desconocida es:", predicted_class)
#print(predicted_class.score)

#En este ejemplo, creamos una instancia del clasificador K-NN con k = 3
#y lo ajustamos a nuestro conjunto de datos X e y.
#Luego, definimos una flor desconocida unknown_flower
#y usamos knn.predict para predecir su clase en función de la clase mayoritaria
#de sus 3 vecinos más cercanos.
#La salida es la clase predicha para la flor desconocida.
