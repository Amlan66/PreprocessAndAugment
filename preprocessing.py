import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
import string
import random
import html

# Download required NLTK data


def preprocess_text(text, tokenize=False, remove_punct=False, add_padding=False):
    result = text
    
    if remove_punct:
        result = result.translate(str.maketrans("", "", string.punctuation))
    
    if tokenize:
        tokens = word_tokenize(result)
        # Generate random colors for different tokens
        unique_tokens = list(set(tokens))
        colors = {token: f"#{random.randint(0, 0xFFFFFF):06x}" for token in unique_tokens}
        
        # Create colored HTML spans for each token
        colored_tokens = [
            f'<span style="color: {colors[token]}">{html.escape(token)}</span>'
            for token in tokens
        ]
        result = " ".join(colored_tokens)
    
    if add_padding:
        result = f"[START] {result} [END]"
    
    return result 