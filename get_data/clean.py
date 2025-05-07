import re

# Comprehensive Latin to Cyrillic mapping for Kazakh
LATIN_TO_CYRILLIC = {
    # Basic Latin (official 2017 Kazakh Latin alphabet equivalents)
    'A': 'А', 'a': 'а',
    'Á': 'Ә', 'á': 'ә',
    'B': 'Б', 'b': 'б',
    'D': 'Д', 'd': 'д',
    'E': 'Е', 'e': 'е',
    'F': 'Ф', 'f': 'ф',
    'G': 'Г', 'g': 'г',
    'Ǵ': 'Ғ', 'ǵ': 'ғ',
    'H': 'Һ', 'h': 'һ',
    'I': 'І', 'i': 'і',
    'Í': 'И', 'í': 'и',
    'J': 'Ж', 'j': 'ж',
    'K': 'К', 'k': 'к',
    'Q': 'Қ', 'q': 'қ',
    'L': 'Л', 'l': 'л',
    'M': 'М', 'm': 'м',
    'N': 'Н', 'n': 'н',
    'Ń': 'Ң', 'ń': 'ң',
    'O': 'О', 'o': 'о',
    'Ó': 'Ө', 'ó': 'ө',
    'P': 'П', 'p': 'п',
    'R': 'Р', 'r': 'р',
    'S': 'С', 's': 'с',
    'T': 'Т', 't': 'т',
    'U': 'Ұ', 'u': 'ұ',
    'Ú': 'Ү', 'ú': 'ү',
    'V': 'В', 'v': 'в',
    'Y': 'Й', 'y': 'й',
    'Ý': 'Ы', 'ý': 'ы',
    'Z': 'З', 'z': 'з',
    
    # Common alternative representations
    'Ä': 'Ә', 'ä': 'ә',
    'Ö': 'Ө', 'ö': 'ө',
    'Ü': 'Ү', 'ü': 'ү',
    'Ū': 'Ұ', 'ū': 'ұ',
    'Ñ': 'Ң', 'ñ': 'ң',
    'Ğ': 'Ғ', 'ğ': 'ғ',
    'Ş': 'Ш', 'ş': 'ш',
    'Ç': 'Ч', 'ç': 'ч',
    'İ': 'І', 'ı': 'ы',
}

KAZAKH_CYRILLIC = "АаӘәБбВвГгҒғДдЕеЁёЖжЗзИиЙйКкҚқЛлМмНнҢңОоӨөПпРрСсТтУуҰұҮүФфХхҺһЦцЧчШшЩщЪъЫыІіЬьЭэЮюЯя"
ALLOWED = f"{KAZAKH_CYRILLIC}0-9\\s.,!?;:\"'«»…\\-—–()_"

def has_high_latin(text: str, threshold=0.2) -> bool:
    """Check if Latin characters exceed threshold ratio"""
    total_chars = len(re.findall(r'\S', text))  # Non-whitespace characters
    if total_chars == 0:
        return False
    
    latin_chars = len(re.findall(r'[A-Za-z]', text))
    return (latin_chars / total_chars) > threshold

def transliterate(text: str) -> str:
    """Convert Latin characters to Kazakh Cyrillic equivalents"""
    return ''.join([LATIN_TO_CYRILLIC.get(c, c) for c in text])

def clean_text(raw: str) -> str:
    """Enhanced cleaning with intelligent transliteration"""
    txt = str(raw).strip()
    
    # Step 1: Handle Latin characters
    if has_high_latin(txt):
        txt = transliterate(txt)  # Convert Latin to Cyrillic
    else:
        txt = re.sub(r'[A-Za-z]', ' ', txt)  # Remove residual Latin
    
    # Step 2: Remove non-Kazakh/non-allowed characters
    txt = re.sub(f'[^{ALLOWED}]', ' ', txt)
    
    # Step 3: Advanced punctuation cleanup
    txt = re.sub(r'\([\s.,!?;:"\'«»…\-—–_]*\)', ' ', txt)  # Parentheses with only punctuation
    txt = re.sub(r'([!?.,:;-])(\s*\1)+', r'\1', txt)       # Collapse spaced punctuation
    txt = re.sub(r'\(\s*\)', ' ', txt)                     # Empty parentheses
    txt = re.sub(r'\s+', ' ', txt)                         # Collapse whitespace
    txt = re.sub(r'([.!?])\1+', r'\1', txt)                # Reduce repeated punctuation
    txt = re.sub(r'([,;:])\s*', r'\1 ', txt)               # Spacing after punctuation
    txt = re.sub(r'\s*([«»])\s*', r'\1', txt)              # Quotation mark cleanup
    txt = re.sub(r'\s*-\s*', ' — ', txt)                   # Format dashes
    
    return txt.strip()