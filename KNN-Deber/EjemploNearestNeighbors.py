from sklearn.neighbors import NearestNeighbors
import numpy as np

#Ejemplo tenemos un conjunto de datos con dos características
#(la longitud y la altura de los pétalos de una flor)
#y queremos encontrar los 3 vecinos más cercanos de un punto de consulta dado:
# 1 Generamos un conjunto de datos de ejemplo
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0], [2, 2]])

# 2 Creamos una instancia de NearestNeighbors, los 3 vecinos mas cecanos.
neigh = NearestNeighbors(n_neighbors=3)

# 3 Ajustamos el modelo a los datos
neigh.fit(X)

# 4 Punto de consulta
query_point = np.array([[2, 2]])

# 5 Encontramos los 3 vecinos más cercanos al punto de consulta
distances, indices = neigh.kneighbors(query_point)

print("Los 3 vecinos más cercanos del punto de consulta son:")
print(X[indices])
print("Las distancias correspondientes son:")
print(distances)

