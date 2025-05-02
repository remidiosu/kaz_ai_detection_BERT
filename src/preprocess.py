import pandas as pd

from typing import Tuple
from get_data.clean import clean_text
from pathlib import Path

def prep_dataset(
        data_dir: Path=Path(__file__).parent.parent / 'data' / 'corpus'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
    """
    1. Load text from corresponding split directories
    2. Clean 
    """
    files = {
        'train': data_dir / 'train' / 'train.csv', 
        'val': data_dir / 'val' / 'val.csv', 
        'test': data_dir / 'test' / 'test.csv'
    }
    # check if files exist
    for split, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} split at {path}")

    # load each split
    dfs = {split:pd.read_csv(path, encoding='utf-8') for split, path in files.items()}

    # clean each text
    for split, df in dfs.items():
        df['text'] = df['text'].map(clean_text)
        df.dropna(subset=['text'], inplace=True)

    return (dfs['train'], dfs['val'], dfs['test'])
    