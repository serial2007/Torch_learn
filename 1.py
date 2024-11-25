from torch.utils.data import Dataset
import os
import pandas as pd # 加载表格
import numpy as np
import torch

class iris_dataloader(Dataset):
	def __init__(self, data_path):
		super().__init__()
		self.data_path = data_path
		assert os.path.exists(self.data_path)
		
		df = pd.read_csv(self.data_path, names=[0,1,2,3,4])
		d = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}
		df[4] = df[4].map(d)

		data = df.iloc[:, 0:4]
		label = df.iloc[:, 4]
		data = (data - np.mean(data) / np.std(data)) # 归一化(Z值化)
		self._data = torch.from_numpy(np.array(data, dtype='float32'))
		self._label = torch.from_numpy(np.array(label, dtype='int64'))
		self._data_num = len(label)

		self.data = list(self._data)
		self.label = list(self._label)

	def __len__(self):
		return self._data_num

	def __getitem__(self, index): # 获取并返回一个数据样本
		return self.data[index], self.label[index]