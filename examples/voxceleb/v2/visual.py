import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path ='/home/jpeng/ntt/work/SPKID/examples/voxceleb/v2/exp/WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40/models/avg_model.pt'
model_weights = torch.load(path,'cpu')
X = model_weights['back_end.att_head.weight'].numpy()
# X = np.random.rand(16, 128)
# X = np.array(...)  # 用你的数据替换这里的...

# 初始化PCA对象，n_components是你想要保留的主成分数量
# 如果你不确定要保留多少个主成分，可以先不设置n_components，然后根据explained_variance_ratio_属性来决定
pca = PCA()  # 你可以设置n_components为一个整数，或者不设置来保留所有成分

# 对数据进行拟合
pca.fit(X)

# 查看累积方差解释率
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# 找到累积方差达到90%的成分数
n_components_90 = np.where(cumulative_variance_ratio >= 0.90)[0][0] + 1

# 绘制累积方差解释率图
plt.figure(figsize=(10, 6), tight_layout=True)
plt.plot(cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.axhline(y=0.90, color='r', linestyle='--')
plt.axvline(x=n_components_90 - 1, color='r', linestyle='--')  # 更新了x的值以正确对齐图表
plt.text(n_components_90, 0.5, f'{n_components_90} Components', color = 'red')
plt.grid(True)

# 保存图像，使用tight布局减少边框间距
plt.savefig('./pca_cumulative_variance.png', bbox_inches='tight', dpi=300)