import pandas as pd
from itertools import islice
from datasets import load_dataset
from get_data.utils import get_yaml_data

def start_download():
    # 1. Load YAML config
    data, root = get_yaml_data('data')
    max_tokens   = data['max_tokens']
    min_tokens   = data['min_tokens']
    sample_count = data.get('count')

    # 2. Dataset & output setup
    dataset_name = "kz-transformers/multidomain-kazakh-dataset"
    domains = ['cc100-monolingual-crawled-data', 'kazakhBooks', 'leipzig', 'oscar', 'kazakhNews']
    cols = ["id", "text", "predicted_language", "contains_kaz_symbols"]

    OUT_DIR = root / "data" / "interim"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for domain in domains:
        print(f"→ Processing {domain} …")
        # 3. Stream the split for this domain
        ds = load_dataset(dataset_name, 
                          data_files=f"{domain}.csv", 
                          split="train",
                          delimiter=',',
                          column_names=cols, 
                          streaming=True)

        # 4. Filter for Kazakh and minimum length
        ds = ds.filter(lambda x: x["predicted_language"] == "kaz")
        ds = ds.filter(lambda x: len(x["text"].split()) >= min_tokens)

        # 5. Shuffle buffer so we can sample randomly
        ds = ds.shuffle(buffer_size=10_000, seed=42)

        # 6. Take only `sample_count` examples
        batch = list(islice(ds, sample_count))

        # 7. Trim each to max_tokens
        processed = [
            {"text": " ".join(item["text"].split()[:max_tokens])}
            for item in batch
        ]

        # 8. Write out CSV with labels
        df = pd.DataFrame(processed)
        df["label"] = 0
        out_path = OUT_DIR / f"{domain}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"   ✔ Saved {len(df)} rows to {out_path}")

    print("Done!")

if __name__ == "__main__":
    start_download()
