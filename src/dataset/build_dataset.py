# download dataset from HuggingFace -> put into .../data/interim
# run only once in the begginning 
from datasets import load_dataset
from tqdm import tqdm
from src.utils import get_yaml_path

import hashlib, json, yaml


## load data YAML
DATA_YAML, root = get_yaml_path('data')
OUT_DIR = root / "data" / "interim"

with DATA_YAML.open('r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

## define helper filters 
def is_kazakh(example):
    return example.get("predicted_language") == "kaz"

def long_enough(example):
    return len(example["text"].split()) >= data['min_tokens']


# the dataset is split across these categories, we will store them in separate json files
out_files = ['cc100-monolingual-crawled-data', 'kazakhBooks', 'leipzig', 'oscar', 'kazakhNews']
cols = ["id", "text", "predicted_language", "contains_kaz_symbols"]

for dom in out_files: 
    print(f"Starting {dom} stream...")
    # compute out file based on category split 
    out_file = OUT_DIR / f"{dom}.jsonl"
    c = 0 

    # init the stream
    stream = load_dataset(
        data["dataset"],               
        data_files=f"{dom}.csv",
        column_names=cols,
        delimiter=",",
        split='train',
        streaming=True
    )

    # open JSON file and start writing into it + filtering
    with out_file.open('w', encoding='utf-8') as f:
        for ex in tqdm(stream):
            if not (is_kazakh(ex) and long_enough(ex)):
                continue
            
            text = ex['text'].replace('\n', ' ')            
            sha1 = hashlib.sha1(text.encode()).hexdigest()
            f.write(
                json.dumps(
                    {"id":sha1, 
                    "text":text}, 
                    ensure_ascii=False
                ) + '\n')
            c += 1

    
    print(f"{dom} split ended! Downloaded article count: {c}")

