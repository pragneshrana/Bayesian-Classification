import numpy as np
import pandas as pd 

x = np.array([[5, 2], [88, 125], [96, 0]]).T
print('np.cov(x): ', np.cov(x))
x=pd.DataFrame([])
a=[5,88,96]
b=[2,125,0]
x["a"]=pd.Series(a)
x["b"]=pd.Series(b)

print('x: ', x)
print('np.cov(x): ', x.cov())
