import nltk
import pandas as pd
import re
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the IMDB dataset (adjust the path as necessary)
df = pd.read_csv('IMDB Dataset.csv') 

# Assuming 'review' contains the text and 'sentiment' labels (positive/negative)
positive_tokens = []
negative_tokens = []

# Set of stopwords to filter out
unwanted_tokens = set(nltk.corpus.stopwords.words('english'))

print("----start----")
# Iterate over the dataset and process each review
for _, row in df.iterrows():
    review = row['review']
    sentiment = row['sentiment']
    
    # Normalize review: lowercasing and removing non-word characters
    cleaned_review = re.sub(r'<.*?>', ' ', review.lower())
     
     # Tokenize the cleaned review
    tokens = word_tokenize(cleaned_review)
     
     # Filter out unwanted tokens
    filtered_tokens = [token for token in tokens if not token.isnumeric() and token not in unwanted_tokens and token not in ["''", '``','*','',',','.','br','<','>','!','(',')','--','?'] ]
     
     # Add tokens to respective lists based on sentiment
    if sentiment == 'positive':
        positive_tokens.extend(filtered_tokens)
    elif sentiment == 'negative':
        negative_tokens.extend(filtered_tokens)


from nltk.util import ngrams
from nltk import FreqDist
# get token_size and vocab_size

token_size = len(positive_tokens) + len(negative_tokens)
print(f"token_size = {token_size}")
token_all = positive_tokens + negative_tokens
token_freq = FreqDist(token_all)
vocab_size = len(token_freq)
print(f"vocab_size = {vocab_size}")

# Assuming positive_tokens is a list of tokenized positive words
bigrams_pos = ngrams(positive_tokens, 2)
bigram_freq_pos = FreqDist(bigrams_pos)
# print(f"bigram_freq_pos.most_common(10):{bigram_freq_pos.most_common(10)}")

bigrams_neg = ngrams(negative_tokens, 2)
bigram_freq_neg = FreqDist(bigrams_neg)
# print(f"bigram_freq_neg.most_common(10):{bigram_freq_neg.most_common(10)}")

# Assuming positive_tokens is a list of tokenized positive words
trigrams_pos = ngrams(positive_tokens, 3)
trigram_freq_pos = FreqDist(trigrams_pos)
# print(f"trigram_freq_pos.most_common(10):{trigram_freq_pos.most_common(10)}")

trigrams_neg = ngrams(negative_tokens, 3)
trigram_freq_neg = FreqDist(trigrams_neg)
# print(f"trigram_freq_neg.most_common(10):{trigram_freq_neg.most_common(10)}")

from nltk import ngrams, FreqDist
import math
def get_freq_prob(word1, word2, word3):
    bigram = list(ngrams(token_all, 2))
    trigram = list(ngrams(token_all, 3))
    word1_word2_count = FreqDist(bigram)[word1, word2]
    word1_word2_word3_count = FreqDist(trigram)[word1,word2,word3]
    res = (1 + word1_word2_word3_count)/(vocab_size + word1_word2_count)
    print(f'{word3}|{word1},{word2}={word1_word2_count},{word1_word2_word3_count},{res}')


# Print the trigram probabilities
get_freq_prob('based', 'true', 'story')