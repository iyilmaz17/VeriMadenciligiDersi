import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd

class KMeans:
    def __init__(self, k=3):
        self.k = k  # Küme sayısı
        self.centroids = None  # Ağırlık merkezlerinin tutulacağı değişken

    @staticmethod
    def euclidean_distance(data_point, centroids):
        # İki nokta arasındaki öklid uzaklığını hesaplayan metot
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        # Verilen veri setine KMeans algoritmasını uygulayan metot
        # X: Veri seti
        # max_iterations: Maksimum iterasyon sayısı

        # Başlangıçta rastgele centroid'leri belirle
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
            y = []

            # Her veri noktasını en yakın Ağırlık merkezin'e atayarak etiketler
            for data_point in X:
                distances = KMeans.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []

            # Her küme için veri noktalarının indislerini topla
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            # Her küme için yeni ağırlık merkezini hesapla
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            # Ağırlık merkezlerinin değişimini kontrol et ve eğer değişim çok küçükse döngüyü sonlandır.
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y

# Blobs veri kümesi oluştur
data = make_blobs(n_samples=100, n_features=2)

# Oluşturulan veri kümesini kullanarak KMeans algoritmasını uygula
random_points = data[0]
excel_data = pd.read_excel('your_excel_file.xlsx')  # 'your_excel_file.xlsx' dosya adınıza göre değiştirin

# Excel verisini Numpy dizisine dönüştür
excel_points = excel_data.to_numpy()
means = KMeans(k=3)
labels = means.fit(random_points)

# Sonuçları görselleştir
plt.scatter(random_points[:, 0], random_points[:, 1], c=labels)
plt.scatter(means.centroids[:, 0], means.centroids[:, 1], c=range(len(means.centroids)),
            marker="*", s=200)
plt.show()
