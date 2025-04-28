import pandas as pd

df2 = pd.read_csv('data/corpus/train/train_human.csv')

dfs = []
parts = []
for split in ['train', 'val', 'test']:
    ps = pd.read_csv(f'data/corpus/{split}/{split}_ai_partial.csv')

    parts.append(ps)

    # print(ps['prompt'].value_counts())

test = parts[-1]
print(test)
print(test['model'].value_counts())

