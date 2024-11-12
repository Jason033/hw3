import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

# 定義高斯分布函數
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# 生成1000筆 x1, x2 的資料
np.random.seed(0)
x1 = np.random.normal(0, np.sqrt(10), 1000)
x2 = np.random.normal(0, np.sqrt(10), 1000)
z = gaussian_function(x1, x2)
data = np.column_stack((x1, x2, z))

# Streamlit app
st.title("3D Gaussian Data Classification with SVM")
st.write("Adjust the threshold and bias to classify points.")

# 使用者設定閾值和偏差
threshold = st.slider("Threshold for Distance", min_value=0.0, max_value=5.0, value=2.5, step=0.1)


# 計算距離並應用偏差
distance = np.sqrt(x1**2 + x2**2)
# 標記資料
labels = np.where(distance < threshold, 0, 1)

# 使用SVM進行分類 (使用三維資料)
svm_model = SVC(kernel='linear')
svm_model.fit(data, labels)

# 獲取係數和截距來繪製分離平面
coef = svm_model.coef_[0]
intercept = svm_model.intercept_

# 3D 可視化
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(121, projection='3d')

# 顏色標記不同類別
colors = np.where(labels == 0, 'blue', 'red')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, marker='o', alpha=0.5)

# 生成網格並計算分離平面的 z 值
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10),
                     np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]  # 使用三維平面方程

# 繪製分離平面
ax.plot_surface(xx, yy, zz, color='black', alpha=0.3)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Z")
ax.set_title("3D Classification Visualization")

# 2D 剖面圖
ax2 = fig.add_subplot(122)
ax2.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
ax2.contourf(xx, yy, zz, levels=[threshold - 0.01, threshold + 0.01], cmap="Greys", alpha=0.2)
ax2.set_xlabel("X1")
ax2.set_ylabel("X2")
ax2.set_title("2D Feature Plane Section")

st.pyplot(fig)
