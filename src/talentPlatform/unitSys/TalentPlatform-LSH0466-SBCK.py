# -*- coding: utf-8 -*-

###############
## Libraries ##
###############

import numpy as np
import scipy.stats as sc
# from SBCK
# from .tools.__Dist import _Dist

# git clone https://github.com/yrobink/SBCK-python.git
import SBCK
import scipy
import matplotlib.pyplot as plt
import xarray as xr
import xclim as xc


###########
## Class ##
###########

## Start with a reference / biased dataset, noted Y,X, from normal distribution:
X = np.random.normal(loc=0, scale=2, size=1000)
Y = np.random.normal(loc=5, scale=0.5, size=1000)

X_da = xr.DataArray(X, dims=['time'], coords={'time': np.arange(0, 1000)})
Y_da = xr.DataArray(Y, dims=['time'], coords={'time': np.arange(0, 1000)})

annual_max_tas = xc.indices.tg_max(X_da, freq='YS')

print(annual_max_tas)



## Generally, we do not know the distribution of X and Y, and we use the empirical quantile mapping:
qm_empiric = SBCK.QM(distY0=SBCK.tools.rv_histogram, distX0=SBCK.tools.rv_histogram)  ## = QM(), default
qm_empiric.fit(Y, X)
Z_empiric = qm_empiric.predict(X)  ## Z is the correction in a non parametric way

## But we can know that X and Y follow a Normal distribution, without knowing the parameters:
qm_normal = SBCK.QM(distY0=scipy.stats.norm, distX0=scipy.stats.norm)
qm_normal.fit(Y, X)
Z_normal = qm_normal.predict(X)

## And finally, we can know the law of Y, and it is usefull to freeze the distribution:
qm_freeze = SBCK.QM(distY0=scipy.stats.norm(loc=5, scale=0.5), distX0=scipy.stats.norm)
qm_freeze.fit(Y, X)  ## = qm_freeze.fit(None,X) because Y is not used
Z_freeze = qm_freeze.predict(X)


# 데이터 생성
X = np.random.normal(loc=0, scale=2, size=1000)
Y = np.random.normal(loc=5, scale=0.5, size=1000)

# QM 맵핑
# ... (위에 제공된 코드 조각으로 계속됩니다.)

# 시각화
plt.figure(figsize=(12, 8))

# 원래 데이터 플롯
plt.subplot(2, 2, 1)
plt.hist(X, bins=30, alpha=0.6, color='g', label='X')
plt.hist(Y, bins=30, alpha=0.6, color='b', label='Y')
plt.title('Original Distributions')
plt.legend()

# Empirical QM 적용 후
plt.subplot(2, 2, 2)
plt.hist(Z_empiric, bins=30, alpha=0.6, color='r', label='Z_empiric')
plt.title('Empirical QM')
plt.legend()

# Normal QM 적용 후
plt.subplot(2, 2, 3)
plt.hist(Z_normal, bins=30, alpha=0.6, color='c', label='Z_normal')
plt.title('Normal QM')
plt.legend()

# Freeze QM 적용 후
plt.subplot(2, 2, 4)
plt.hist(Z_freeze, bins=30, alpha=0.6, color='m', label='Z_freeze')
plt.title('Freeze QM')
plt.legend()

# 전체 그래프 보여주기
plt.tight_layout()
plt.show()