import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

import itertools
import argparse
import re
import os
import pickle
import math


def get_tradeview_btc_data():
    data = pd.read_csv(r'C:\Users\joshu\Documents\RL\tadebot\tradeview_bitcoin_1d.csv')
    price_volume_data = data.drop(['time', 'open', 'high', 'low'], axis=1)
    return price_volume_data.values

if __name__ == '__main__':

    data = get_tradeview_btc_data()
    print(data)
    print(data.shape)