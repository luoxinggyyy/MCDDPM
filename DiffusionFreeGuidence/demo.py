import numpy as np
import torch
import pandas as pd
from plotnine import ggplot, aes, geom_line,geom_point
a = np.load('./show/e_69.npy')

## 
# np_to_csv = pd.DataFrame(data = a)
# np_to_csv.to_csv('np_to_csv.csv'
# )
b = pd.read_csv('cc3.csv').dropna()
print(b.size)

(
    ggplot(b, aes(x='real_por', y='test_por'))
    + geom_point()
    + geom_abline(intercept=45,
                  slope=-5,
                  color='blue',      # set line colour
                  size=2,            # set line thickness
                  linetype="dashed"  # set line type
                 )
    + labs(x='real_por', y='test_por')
)

