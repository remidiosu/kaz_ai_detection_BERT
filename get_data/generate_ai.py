# scripts/generate_ai.py

import os
import time
import random
import logging
from pathlib import Path
from typing import List

import pandas as pd
import openai

from get_data.utils import get_yaml_data


def generate_paraphrases() -> None:
    # Load config
    cfg, ROOT = get_yaml_data('generate')
    openai.api_key = cfg.get('api_key') or os.getenv('OPENAI_API_KEY')
    corpus_dir = ROOT / 'data' / 'corpus'
    splits: List[str] = cfg.get('splits', ['train', 'val', 'test'])
    rate_sleep = cfg.get('rate_limit_sleep', 1.0)

    # Setup logging
    log_dir = ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'generate_ai.log'
    logging.basicConfig(
        level=logging.INFO,
        filename=log_path,
        format='%(asctime)s %(levelname)s %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting AI paraphrase generation")

    for split in splits:
        human_path = corpus_dir / split / f"{split}_human.csv"
        ai_path    = corpus_dir / split / f"{split}_ai.csv"

        if not human_path.exists():
            logger.warning(f"[{split}] human CSV not found at {human_path}, skipping")
            continue

        df_human = pd.read_csv(human_path, encoding='utf-8')
        records = []

        for idx, row in df_human.iterrows():
            text = row['text']
            model = random.choice(cfg['models'])
            temperature = random.choice(cfg['temperatures'])
            prompt_tpl = random.choice(cfg['prompts'])
            prompt = prompt_tpl.format(text=text)

            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                ai_text = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"[{split}][idx={idx}] OpenAI error: {e}")
                ai_text = ""

            records.append({
                'original_text': text,
                'text': ai_text,
                'label': 1,
                'model': model,
                'temperature': temperature,
                'prompt': prompt,
            })

            time.sleep(rate_sleep)

        # Save out AI paraphrases
        df_ai = pd.DataFrame.from_records(records)
        df_ai.to_csv(ai_path, index=False, encoding='utf-8')
        logger.info(f"[{split}] saved {len(df_ai)} paraphrases to {ai_path}")

    logger.info("Finished AI paraphrase generation")


if __name__ == '__main__':
    generate_paraphrases()
