# coding: utf-8
import numpy as np
import pandas as pd
if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    df = pd.read_json('./data/ag1608_20160812.jcsv')
    print df
