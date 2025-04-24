import os
import time
import random
import logging
from typing import List, Dict

import pandas as pd
import openai
from openai import RateLimitError

from get_data.utils import get_yaml_data


def generate_paraphrases() -> None:
    # Load config
    cfg, ROOT = get_yaml_data('generate')
    openai.api_key = cfg.get('api_key') or os.getenv('OPENAI_API_KEY')

    corpus_dir = ROOT / 'data' / 'corpus'
    splits: List[str] = cfg.get('splits', ['train', 'val', 'test'])
    rate_sleep = cfg.get('rate_limit_sleep', 1.0)

    # Model selection weights
    models: List[str] = cfg['models']           
    weights: List[float] = cfg['weights']       
    temperatures: List[float] = cfg.get('temperatures', [0.7])
    temperature_weights: List[float] = cfg.get('t_weights')
    prompts: List[str] = cfg['prompts']
    max_tokens: int = cfg.get('max_completion_tokens', 1500)

    # Setup logging
    log_dir = ROOT / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'generate_ai.log'
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting AI paraphrase generation")

    for split in splits:
        human_csv = corpus_dir / split / f"{split}_human.csv"
        combined_csv = corpus_dir / split / f"{split}.csv"

        if not human_csv.exists():
            logger.warning(f"[{split}] human CSV not found at {human_csv}, skipping")
            continue
        if combined_csv.exists():
            logger.info(f"[{split}] combined CSV already exists at {combined_csv}, skipping generation")
            continue

        # Read human data
        df_human = pd.read_csv(human_csv, encoding='utf-8')
        df_human = df_human[['text', 'label']].copy()

        records: List[Dict] = []
        for idx, text in enumerate(df_human['text']):
            # Weighted random model selection
            model = random.choices(models, weights=weights, k=1)[0]
            temperature = random.choices(temperatures, temperature_weights, k=1)[0]
            prompt = random.choice(prompts).format(text=text)

            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                ai_text = resp.choices[0].message.content.strip()
            except RateLimitError:
                logger.warning(f"[{split}][idx={idx}] rate limit hit, sleeping 20s")
                time.sleep(20)
                continue
            except Exception as e:
                logger.error(f"[{split}][idx={idx}] OpenAI error: {e}")
                continue

            if not ai_text:
                logger.warning(f"[{split}][idx={idx}] empty response, skipping")
                continue

            records.append({
                'text': ai_text,
                'label': 1,
                'model': model,
                'temperature': temperature,
                'prompt': prompt
            })

            # Respect rate limits
            time.sleep(rate_sleep)

        # Build AI DataFrame
        df_ai = pd.DataFrame.from_records(records)
        df_ai = df_ai[['text', 'label']]

        # Combine human + AI and save
        df_human['label'] = 0
        df_combined = pd.concat([df_human, df_ai], ignore_index=True)

        # shuffle combined 
        df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
        df_combined.to_csv(combined_csv, index=False, encoding='utf-8')

        logger.info(f"[{split}] saved combined {len(df_combined)} rows to {combined_csv}")

    logger.info("Finished AI paraphrase generation")


if __name__ == '__main__':
    generate_paraphrases()
