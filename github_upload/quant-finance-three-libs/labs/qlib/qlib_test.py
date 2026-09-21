import numpy as np
import lightgbm as lgb
import qlib
from qlib.data._libs.expanding import expanding_mean
from qlib.data._libs.rolling import rolling_mean


values = np.arange(1, 7, dtype=float)

features = np.arange(20, dtype=float).reshape(10, 2)
labels = features[:, 0] * 0.5 + features[:, 1] * 0.2
model = lgb.LGBMRegressor(n_estimators=5, verbosity=-1)
model.fit(features, labels)

print("Qlib 版本：", qlib.__version__)
print("滚动均值：", rolling_mean(values, 3))
print("扩展均值：", expanding_mean(values))
print("LightGBM 预测：", model.predict(features[:2]))
print("Qlib 测试成功")
