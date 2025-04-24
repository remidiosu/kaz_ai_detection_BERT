import re

# Kazakh Cyrillic block, including ӘғҚңӨұҮҺІ
KAZAKH_CHARS = "А-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі"
ALLOWED = f"{KAZAKH_CHARS}0-9\\s.,!?;:\"'«»…\\-—–()"

NOT_KAZAKH_RE = re.compile(f"[^{ALLOWED}]+")

def clean_text(raw: str) -> str:
    """
    Remove every character except Kazakh letters, digits, whitespace, and basic punctuation.
    Collapse runs of whitespace to a single space, and strip leading/trailing spaces.
    """
    txt = str(raw)
    # 1) strip all disallowed chars
    txt = NOT_KAZAKH_RE.sub(" ", txt)
    # 2) collapse whitespace
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()
