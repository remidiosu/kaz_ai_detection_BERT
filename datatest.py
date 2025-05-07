from pathlib import Path
import pandas as pd

data_dir = Path(__file__).parent / 'data' / 'corpus'

tr = data_dir / 'train' / 'train_ai_partial.csv'
te = data_dir / 'test' / 'test_ai_partial.csv'
va = data_dir / 'val' / 'val_ai_partial.csv'

train_df = pd.read_csv(tr)
test_df = pd.read_csv(te)
val_df = pd.read_csv(va)

def normalize_prompt(p):
    return p.replace('\r\n', '\n').replace('\r', '\n').strip()

for df in [train_df, test_df, val_df]:
    df['prompt'] = df['prompt'].apply(normalize_prompt)

# Combine all splits
df_all = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Compute text length
df_all['text_length'] = df_all['text'].str.len()

# 1) Basic count+% summary function
def summarize_counts(df, column):
    counts = df[column].value_counts(dropna=False).reset_index()
    counts.columns = [column, 'Count']
    counts['Percentage (%)'] = round((counts['Count'] / len(df)) * 100, 1)
    return counts

# 2) Length summary function (count + average length + %)
def summarize_with_length(df, group_col):
    summary = (
        df
        .groupby(group_col)
        .agg(
            Count=('text_length', 'count'),
            Avg_Text_Length=('text_length', 'mean')
        )
        .reset_index()
    )
    summary['Percentage (%)'] = round((summary['Count'] / len(df)) * 100, 1)
    return summary.sort_values('Count', ascending=False)

# Prepare short prompts for easy reading
df_all['prompt_short'] = df_all['prompt'].apply(lambda x: x.split('\n')[0][:40] + '...')

# Summaries
model_counts = summarize_counts(df_all, 'model')
temp_counts  = summarize_counts(df_all, 'temperature')
prompt_counts = summarize_counts(df_all, 'prompt_short')

model_length = summarize_with_length(df_all, 'model')
temp_length  = summarize_with_length(df_all, 'temperature')
prompt_length = summarize_with_length(df_all, 'prompt_short')

# Print everything
print("\n=== MODEL USAGE ===")
print(model_counts.to_string(index=False))
print("\n--- model vs. avg text length ---")
print(model_length.to_string(index=False))

print("\n=== TEMPERATURE USAGE ===")
print(temp_counts.to_string(index=False))
print("\n--- temperature vs. avg text length ---")
print(temp_length.to_string(index=False))

print("\n=== PROMPT USAGE ===")
print(prompt_counts.to_string(index=False))
print("\n--- prompt vs. avg text length ---")
print(prompt_length.to_string(index=False))
