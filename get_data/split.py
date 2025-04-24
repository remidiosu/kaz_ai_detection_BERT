# gets the downloaded data from data/intermim, clean and split into train/test/validation and save results in data/corpus

from pathlib import Path
from sklearn.model_selection import train_test_split
from get_data.utils import get_yaml_data
from get_data.clean import clean_text

import os
import pandas as pd

# read downloaded data files into list of docs
def read_interim():
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    data = ROOT / 'data' / 'interim'
    
    dfs = []

    # read all cdv data files in interim 
    for filename in os.listdir(data): 
        if filename.endswith('.csv'):
            filepath = data / filename
            df = pd.read_csv(filepath)
            # clean the text
            df["text"] = df["text"].apply(clean_text)
            dfs.append(df) 

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    return combined


def split():
    # load YAML file for split instructions 
    data, root = get_yaml_data('data') 

    # read interim data
    df = read_interim()
    X = df['text']
    y = df['label']

    # split the data into: test / train
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        X, y, test_size=data['test'], random_state=data.get('seed'), stratify=y
    )

    # further split the train into: train / val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=data['val'], random_state=data.get('seed'), stratify=train_labels
    )

    # dfs for each 
    df_train = pd.DataFrame({'text': train_texts,'label': train_labels})
    df_val = pd.DataFrame({'text': val_texts,'label': val_labels})
    df_test = pd.DataFrame({'text': test_texts,'label': test_labels})
    
    # make sure the correct dirs exist
    for split in ['train', 'val', 'test']:
        (root / 'data' / 'corpus' / split).mkdir(parents=True, exist_ok=True)

    # create csv for: test, train, val 
    df_train.to_csv(f'{root}/data/corpus/train/train_human.csv', index=False, encoding='utf-8')
    df_val.to_csv(f'{root}/data/corpus/val/val_human.csv',   index=False, encoding='utf-8')
    df_test.to_csv(f'{root}/data/corpus/test/test_human.csv',  index=False, encoding='utf-8')

    return (
        f"Created → test: {df_test.shape[0]}, "
        f"train: {df_train.shape[0]}, "
        f"val: {df_val.shape[0]}"
    )








