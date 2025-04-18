# gets the downloaded data from data/intermim into train/test/validation and save results in data/corpus

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import get_yaml_data

import os
import pandas as pd

# read downloaded data files into list of docs
def read_interim():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    data = ROOT / 'data' / 'interim'
    
    dfs = []
    for filename in os.listdir(data): 
        if filename.endswith('.csv'):
            df = pd.read_csv(filename)
            dfs.append(df) 

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined


def split():
    # load YAML file for split instructions 
    data = get_yaml_data('data') 

    # read interim data
    df = read_interim()
    X = df['text']
    y = df['label']

    # split the data into: test / train
    train_texts, train_labels, test_texs, test_labels = train_test_split(
        X, y, test_size=data['test'], random_state=data['seed'] 
    )

    # further split the train into: train / val
    
    # create csv for: test, train, val 
    # cols: text,label


    # save the csv for test in data/test/human

    # save the csv for train, val in data/corpus/human






