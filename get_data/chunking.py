import time
import tiktoken
from openai import OpenAI

def chunk_text(text: str, encoding: tiktoken.Encoding, chunk_size: int, overlap: int):
    """
    Yield successive overlapping chunks of text (as strings), each no longer than chunk_size tokens.
    """
    tokens = encoding.encode(text)
    start = 0
    length = len(tokens)
    while start < length:
        end = min(start + chunk_size, length)
        yield encoding.decode(tokens[start:end])
        start += chunk_size - overlap


def paraphrase_long_text(
    text: str,
    client: OpenAI,
    prompt_template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    encoding: tiktoken.Encoding,
    context_limit: int,
    margin: int,
    overlap: int,
    rate_sleep: float
) -> str:
    """
    Split `text` into manageable chunks, paraphrase each via the OpenAI client,
    and reassemble the outputs into one string.
    """
    # Compute header token count
    header = prompt_template.format(text="")
    header_tokens = encoding.encode(header)
    # Available tokens for each chunk
    available = context_limit - max_tokens - len(header_tokens) - margin

    paraphrased = []
    for chunk in chunk_text(text, encoding, available, overlap):
        prompt = prompt_template.format(text=chunk)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        paraphrased.append(resp.choices[0].message.content.strip())
        time.sleep(rate_sleep)

    # Join chunks into one output
    return "\n".join(paraphrased)