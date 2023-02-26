# Libreria para Presentacion de Grafica
import matplotlib.pyplot as plt

# Declaracion de la Variables Arreglo DATASET para ver el BIG(o)
n_values = [10000, 20000, 30000, 40000, 50000]
t_values = [0.00698, 0.00798, 0.00898, 0.00997, 0.01097]

plt.plot(n_values, t_values)
plt.xlabel(' Datos del Arreglo <DataSet>')
plt.ylabel(' Linea de Tiempo en Segundos')
plt.title(' Comportamiento BIG(O) - Algoritmo de Crecimiento en Eficiencia Lineal')
plt.show()