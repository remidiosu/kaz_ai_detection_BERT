import os
import random
import logging
import pandas as pd
import tiktoken

from openai import OpenAI
from dotenv import load_dotenv
from get_data.utils import get_yaml_data
from get_data.chunking import paraphrase_long_text


def generate_paraphrases() -> None:
    # Load config
    load_dotenv()
    cfg, ROOT = get_yaml_data('generate')
    api_key = cfg.get('api_key') or os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    corpus_dir = ROOT / 'data' / 'corpus'
    splits = ['train', 'val', 'test']
    rate_sleep = cfg.get('rate_limit_sleep', 1.0)

    models = cfg['models']
    weights = cfg['weights']
    temperatures = cfg.get('temperatures', [0.7])
    temp_weights = cfg.get('tempweights')
    prompts = cfg['prompts']
    max_tokens = cfg.get('max_completion_tokens', 1500)
    overlap = cfg.get('overlap_tokens', 200)

    # Tokenizer & context setup
    encoding = tiktoken.encoding_for_model(models[0])
    CONTEXT_LIMIT = 8192
    MARGIN = 50

    # Logging
    log_dir = ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=log_dir / 'generate_ai.log',
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)

    for split in splits:
        human_csv = corpus_dir / split / f"{split}_human.csv"
        combined_csv = corpus_dir / split / f"{split}.csv"
        partial_csv = corpus_dir / split / f"{split}_ai_partial.csv"

        if not human_csv.exists():
            logger.warning(f"[{split}] human CSV not found, skipping")
            continue
        if combined_csv.exists():
            logger.info(f"[{split}] combined CSV exists, skipping")
            continue

        # Read human data
        df_human = pd.read_csv(human_csv, encoding='utf-8')[['text', 'label']].copy()

        # Resume partial paraphrases
        if partial_csv.exists():
            df_prev = pd.read_csv(partial_csv, encoding='utf-8')
            records = df_prev.to_dict(orient='records')
            processed = len(records)
        else:
            records = []
            processed = 0

        # Generate paraphrases
        for idx, text in enumerate(df_human['text']):
            if idx < processed:
                continue

            model = random.choices(models, weights=weights, k=1)[0]
            temperature = random.choices(temperatures, weights=temp_weights, k=1)[0]
            prompt_template = random.choice(prompts)

            try:
                ai_text = paraphrase_long_text(
                    text=text,
                    client=client,
                    prompt_template=prompt_template,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    encoding=encoding,
                    context_limit=CONTEXT_LIMIT,
                    margin=MARGIN,
                    overlap=overlap,
                    rate_sleep=rate_sleep
                )
            except Exception as e:
                logger.error(f"[{split}][idx={idx}] unexpected error: {e}")
                continue

            if not ai_text:
                logger.warning(f"[{split}][idx={idx}] empty response, skipping")
                continue

            record = {
                'text': ai_text,
                'label': 1,
                'model': model,
                'temperature': temperature,
                'prompt': prompt_template
            }
            records.append(record)

            # Append new metadata record
            pd.DataFrame([record]).to_csv(
                partial_csv,
                mode='a',
                header=not partial_csv.exists(),
                index=False,
                encoding='utf-8'
            )

        # Build final df_ai from partials
        df_ai = pd.read_csv(partial_csv, encoding='utf-8') if partial_csv.exists() else pd.DataFrame(records)

        # Combine human + AI and save
        df_human['label'] = 0
        df_combined = pd.concat([df_human, df_ai[['text','label']]], ignore_index=True).sample(frac=1, random_state=42)
        df_combined.to_csv(combined_csv, index=False, encoding='utf-8')

        logger.info(f"[{split}] saved {df_combined.shape[0]} rows.")

if __name__ == '__main__':
    generate_paraphrases()
