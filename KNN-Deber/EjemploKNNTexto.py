from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Cargar los datos desde un archivo CSV
data = pd.read_csv('DataSet/nombres.csv')

# Definir los datos de entrada para el modelo
X = data[['nombre']]

# Definir el modelo KNN con k=5
knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')

# Entrenar el modelo con los datos de entrada
knn.fit(X.to_numpy())

# Realizar la búsqueda de nombres similares
nombre_busqueda = 'Jose'
busqueda = knn.kneighbors([[nombre_busqueda]], return_distance=False)

# Imprimir los resultados de la búsqueda
resultados = data.iloc[busqueda[0]]
print(resultados)
