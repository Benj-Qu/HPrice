from joblib import load
import numpy as np
import pandas as pd 
import os



class Model():
	def __init__(self, dir_name: str) -> None:
		self.dir = dir_name

	def myPredict(self, data: pd.DataFrame) -> np.ndarray:
		m = load(os.path.join(self.dir, '519370910113-2.gz'))
		return m.predict(data) * data['Building Square Feet']
