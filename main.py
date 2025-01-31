import matplotlib
matplotlib.use('TkAgg') # Ustawia backend Matplotlib na "TkAgg" (do wizualizacji).
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier # Biblioteczny KNN.
from sklearn.datasets import make_classification # Generowanie danych testowych.
from sklearn.model_selection import train_test_split # Podział danych na treningowe i testowe.
from sklearn.metrics import accuracy_score # Do oceny dokładności klasyfikatora.
import matplotlib.pyplot as plt  
import time # Do mierzenia czasu wykonania.
import unittest # Do pisania testów jednostkowych.


# Implementacja algorytmu Nearest Neighbours
class NearestNeighbours:
    def __init__(self, k=3):
        self.k = k                # Liczba najbliższych sąsiadów do wzięcia pod uwagę.
        self.data = None          # Przechowuje dane treningowe.
        self.labels = None        # Przechowuje etykiety klas.

    def fit(self, data, labels):    # Funkcja trenująca
        self.data = np.array(data)
        self.labels = np.array(labels)

    def _euclidean_distance(self, point1, point2):    # Obliczanie odległości euklidesowej
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, test_data):    # Przewidywanie klasy dla danych testowych
        predictions = []
        for test_point in test_data:
            distances = []
            for i, train_point in enumerate(self.data):
                dist = self._euclidean_distance(test_point, train_point)
                distances.append((dist, self.labels[i]))
            distances.sort(key=lambda x: x[0])
            nearest_labels = [label for _, label in distances[:self.k]]
            most_common = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)


# Generowanie danych testowych
def generate_data():    # Generowanie danych sztucznych.
    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42)


# Porównanie działania algorytmu własnego i bibliotecznego
def compare_algorithms():
    X_train, X_test, y_train, y_test = generate_data()

    # Implementacja własna
    start_time = time.time()
    nn = NearestNeighbours(k=5)
    nn.fit(X_train, y_train)
    custom_predictions = nn.predict(X_test)
    custom_time = time.time() - start_time
    custom_accuracy = accuracy_score(y_test, custom_predictions)

    # Wersja biblioteczna
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    library_predictions = knn.predict(X_test)
    library_time = time.time() - start_time
    library_accuracy = accuracy_score(y_test, library_predictions)

    print("Custom Implementation:")
    print(f"Accuracy: {custom_accuracy:.2f}, Time: {custom_time:.4f} seconds")
    print("Library Implementation:")
    print(f"Accuracy: {library_accuracy:.2f}, Time: {library_time:.4f} seconds")

    # Wizualizacja wyników
    labels = ['Custom', 'Library']
    times = [custom_time, library_time]
    accuracies = [custom_accuracy, library_accuracy]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.bar(labels, times, color=['blue', 'green'])
    plt.title('Comparison of Execution Time')
    plt.ylabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    plt.bar(labels, accuracies, color=['blue', 'green'])
    plt.title('Comparison of Accuracy')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()

    # Wizualizacja dopasowań najbliższych sąsiadów
    plot_nearest_neighbors(X_train, y_train, X_test, y_test, nn)

    # Wizualizacja punktów testowych i ich klasyfikacji
    plot_test_classification(X_train, y_train, X_test, y_test, nn)


# Funkcja do wizualizacji dopasowań najbliższych sąsiadów
def plot_nearest_neighbors(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(8, 8))

    # Rysowanie danych treningowych
    for label in np.unique(y_train):
        points = X_train[y_train == label]
        plt.scatter(points[:, 0], points[:, 1], label=f"Class {label}", alpha=0.6)

    # Rysowanie danych testowych z dopasowaniami
    predictions = model.predict(X_test)
    for i, point in enumerate(X_test):
        plt.scatter(point[0], point[1], c='red' if predictions[i] == y_test[i] else 'black', marker='x')

    plt.title("Nearest Neighbors Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()


# Funkcja do wizualizacji punktów testowych i ich klasyfikacji
def plot_test_classification(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(8, 8))

    # Rysowanie danych testowych z wynikami klasyfikacji
    predictions = model.predict(X_test)
    for label in np.unique(y_train):
        points = X_test[predictions == label]
        plt.scatter(points[:, 0], points[:, 1], label=f"Predicted Class {label}", alpha=0.6, edgecolor='k')

    # Rysowanie danych treningowych
    for label in np.unique(y_train):
        points = X_train[y_train == label]
        plt.scatter(points[:, 0], points[:, 1], label=f"Training Class {label}", alpha=0.3)

    plt.title("Test Points Classification")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid()
    plt.show()


# Unittesty klasy NearestNeighbours
class TestNearestNeighbours(unittest.TestCase):
    def test_prediction(self):
        data = [[1, 1], [2, 2], [3, 3]]
        labels = [0, 1, 1]
        test_data = [[1.5, 1.5], [3.1, 3.1]]
        expected = [1, 1]

        nn = NearestNeighbours(k=3)
        nn.fit(data, labels)
        predictions = nn.predict(test_data)

        self.assertTrue(np.array_equal(predictions, expected))


#if __name__ == "__main__":
    compare_algorithms()
    #unittest.main(argv=[''], exit=False)
