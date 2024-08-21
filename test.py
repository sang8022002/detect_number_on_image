# from sklearn.cluster import DBSCAN
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import filedialog

# # Tạo dữ liệu mẫu và áp dụng DBSCAN
# X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# labels = db.labels_

# # Vẽ kết quả
# plt.scatter(X[:,0], X[:,1], c=labels, cmap='rainbow', alpha=0.7, edgecolors='b')
# plt.title('DBSCAN Clustering')
# plt.show()

# # Sau khi vẽ xong, mở cửa sổ chọn file
# root = tk.Tk()
# root.withdraw() # Ẩn cửa sổ tkinter chính
# file_path = filedialog.askopenfilename() # Mở cửa sổ chọn file và lấy đường dẫn file được chọn
# print(f"File selected: {file_path}")

# # use K means clustering with data is generated from make_blobs
# from sklearn.cluster import KMeans
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Create data
# X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# # Standardize data
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)

# # Apply KMeans
# kmeans = KMeans(n_clusters=4, random_state=0)
# kmeans.fit(X_std)
# y_kmeans = kmeans.predict(X_std)

# # Visualize result
# plt.scatter(X_std[:, 0], X_std[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
# plt.title('KMeans Clustering')
# plt.show()

from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Tạo dữ liệu
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Áp dụng Mean Shift
mean_shift = MeanShift()
mean_shift.fit(X_std)
labels = mean_shift.labels_

# Trực quan hóa kết quả
plt.scatter(X_std[:, 0], X_std[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Mean Shift Clustering')
plt.show()