# -*- coding: utf-8 -*-

from trend import ad
from trend import aroon

import pandas as pd 
import numpy as np

def test_aroon(df):
    indest = df.index
    df_date = pd.Series(df['date'].tolist(),index = indest)
    high = pd.Series(df['high'].tolist(),index = indest)
    low = pd.Series(df['low'].tolist(),index = indest)
    down,up = aroon.AROON.result(high, low)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    df = pd.read_csv('../data/ag1601_2016-1-4_2016-1-15.jcsv', sep = ',' )
    test_aroon(df)
