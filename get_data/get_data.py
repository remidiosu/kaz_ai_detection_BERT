# download dataset from HuggingFace -> put into .../data/interim
# run only once in the begginning 

from get_data.utils import get_yaml_data

from huggingface_hub import hf_hub_download
import pandas as pd
import os

# disable symlinks because of dev setup
# remove on cloud later
# os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


def start_download():
    ## load data YAML
    data, root = get_yaml_data('data')
    OUT_DIR = root / "data" / "interim"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # the dataset is split across these categories, we will store them in separate json files
    out_files = ['cc100-monolingual-crawled-data', 'kazakhBooks', 'leipzig', 'oscar', 'kazakhNews']
    reports = []
    for dom in out_files: 
        # 1. Download the CSV once to disk via Git‑LFS
        file_path = hf_hub_download(
            repo_id="kz-transformers/multidomain-kazakh-dataset",
            filename=f"{dom}.csv",
            repo_type='dataset'
        )

        # 2. Read with pandas (auto handles the header row)
        df = pd.read_csv(file_path)

        # 3. Filter & sample
        df = df[df.predicted_language == "kaz"]
        df['text'] = (df['text'].str.split().str[:data['max_tokens']].str.join(' '))
        df = df[df.text.str.split().str.len() >= data['min_tokens']]
        df = df.sample(n=data['count'], random_state=42)

        # 4. Write out two‑column CSV
        df[["text"]].assign(label=0).to_csv(f"{OUT_DIR}/{dom}.csv", index=False, encoding="utf-8")
        reports.append(df.describe())

    return f"Downloaded dfs {reports}"


