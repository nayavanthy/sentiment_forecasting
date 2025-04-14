import openai
import os
import gensim.downloader as api
import spacy
import re
import concurrent.futures

output_dir = '/home/captain/Desktop/NLP_FISAC/Backend/Hashtag_Generation'

# Load a Higher-Dimensional Pre-trained Word Embedding Model
print("Loading Word Embeddings (GloVe)...")
word2vec_model = api.load("glove-wiki-gigaword-300")  # More accurate embeddings
print("Word Embeddings Loaded!")

# Load spaCy NLP Model
print("Loading spaCy NLP model...")
nlp = spacy.load("en_core_web_md")  # Medium-sized model with word vectors
print("spaCy Model Loaded!")

def save_keywords_to_file(keywords, filename="keywords.txt"):
    with open(os.path.join(output_dir,filename), "w") as file:
        for word in keywords:
            file.write(word+"\n")

def run(topic):
    keywords = generate_hashtags(topic)
    save_keywords_to_file(keywords)
    print(f"keywords saved to keywords.txt")

def get_word2vec_similar_words(word, top_n=10):
    """Fetches similar words from GloVe embeddings."""
    try:
        if word in word2vec_model:
            return [w for w, _ in word2vec_model.most_similar(word, topn=top_n)]
        elif word.lower() in word2vec_model:
            return [w for w, _ in word2vec_model.most_similar(word.lower(), topn=top_n)]
    except KeyError:
        return []
    return []

def extract_relevant_words(text, nlp):
    """Extracts important words using NLP without manual intervention."""
    doc = nlp(text)
    
    # Keep only meaningful nouns and proper nouns
    relevant_words = {token.lemma_ for token in doc if token.pos_ in {"NOUN", "PROPN"}}
    
    return list(relevant_words)  # Return unique words

def filter_hashtag_phrases(phrases):
    """Filters phrases dynamically based on frequency and length."""
    
    # Exclude common generic words using GloVe similarity scores
    filtered_phrases = [
        phrase for phrase in phrases 
        if 2 <= len(phrase) <= 25 and phrase in word2vec_model
    ]
    
    return filtered_phrases

def format_hashtag(text):
    """Formats text into a proper hashtag."""
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.strip())  # Remove special characters
    words = text.split()
    
    if not words:
        return None

    return "#" + "".join(word.capitalize() for word in words)

def generate_hashtags(topic, max_hashtags=30):
    """Generates meaningful hashtags for a given topic."""
    print(f"Generating hashtags for: {topic}\n")
    
    # Extract important words (nouns, proper nouns) from the topic
    relevant_words = extract_relevant_words(topic, nlp)
    relevant_words.append(topic)  # Ensure the original topic is included
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_w2v = executor.map(get_word2vec_similar_words, relevant_words)
        
    similar_words = set()
    for word_list in future_w2v:
        similar_words.update(word_list)
    
    # Filter and format hashtags
    all_phrases = filter_hashtag_phrases(similar_words)
    
    return all_phrases[:max_hashtags]

if __name__ == "__main__":
    input_word = input("Enter a topic: ")
    hashtags = generate_hashtags(input_word)
    print("\nGenerated Hashtags:")
    for tag in hashtags:
        print(tag)

    save_keywords_to_file(hashtags)

