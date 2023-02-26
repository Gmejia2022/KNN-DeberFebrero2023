import math

class NearestNeighbors:
    def __init__(self):
        self._data = []
        self._labels = []

    def add_data(self, data, label):
        self._data.append(data)
        self._labels.append(label)

    def find_neighbors(self, k, query):
        distances = []
        for i in range(len(self._data)):
            distance = math.sqrt(sum([(query[j] - self._data[i][j])**2 for j in range(len(query))]))
            distances.append((distance, self._labels[i]))
        distances.sort()

        neighbors = [distances[i][1] for i in range(k)]
        return neighbors

nn = NearestNeighbors()
nn.add_data([5.1, 3.5, 1.4, 0.2], "Gonzalo")
nn.add_data([4.9, 3.0, 1.4, 0.2], "Valentina")
nn.add_data([6.2, 3.4, 5.4, 2.3], "Nancy")
nn.add_data([5.0, 3.0, 4.8, 1.8], "Eugenio")
nn.add_data([3.0, 3.0, 1.8, 1.8], "Carlos")
nn.add_data([2.0, 3.8, 2.8, 1.8], "Xavier")
nn.add_data([4.0, 3.1, 5.8, 1.8], "Paul")

query = [5.1, 3.5, 1.4, 0.2]
k = 4
result = nn.find_neighbors(k, query)
for label in result:
    print("Vecino cercano:", label)