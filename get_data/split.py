# gets the downloaded data from data/intermim into train/test/validation and save results in data/corpus

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import get_yaml_data

import os
import pandas as pd

# read downloaded data files into list of docs
def read_interim() -> list:
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    data = ROOT / 'data' / 'interim'
    
    for filename in os.listdir(data): 
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
             


def split():
    # load YAML file for split instructions 
    data = get_yaml_data('data') 

    # read interim data
    docs = read_interim()


    # split the data into: test / train

    # further split the train into: train / val

    # create csv for: test, train, val 
    # cols: text,label


    # save the csv for test in data/test/human

    # save the csv for train, val in data/corpus/human






