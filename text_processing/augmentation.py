import nltk
from nltk.corpus import wordnet
import random
import html

nltk.download('wordnet')

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(set(synonyms))

def augment_text(text, synonym_replace=False, random_insert=False):
    words = text.split()
    result = words.copy()
    
    if synonym_replace:
        # Keep track of changed words for coloring
        changed_words = {}
        
        for i, word in enumerate(words):
            synonyms = get_synonyms(word)
            if synonyms:
                new_word = random.choice(synonyms)
                changed_words[i] = (word, new_word)
                result[i] = new_word
        
        # Color the changed words
        for i, (old_word, new_word) in changed_words.items():
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            result[i] = f'<span style="color: {color}" title="{html.escape(old_word)}">{html.escape(new_word)}</span>'
    
    if random_insert:
        # Insert random words from vocabulary
        vocab = list(set(words))
        num_insertions = len(words) // 4  # Insert 25% more words
        
        for _ in range(num_insertions):
            insert_pos = random.randint(0, len(result))
            insert_word = random.choice(vocab)
            color = f"#{random.randint(0, 0xFFFFFF):06x}"
            result.insert(insert_pos, f'<span style="color: {color}">{html.escape(insert_word)}</span>')
    
    return " ".join(result) 