import pandas as pd

df2 = pd.read_csv('data/corpus/train/train_human.csv')

dfs = []
parts = []
for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/corpus/{split}/{split}.csv')
    print(df['label'].value_counts())
    dfs.append(df)




