import nltk
import re
import numpy as np
import pandas as pd
#Tokenization of text
from nltk.tokenize import word_tokenize,sent_tokenize
#remove stop-words
from nltk.corpus import stopwords # library
nltk.download('stopwords')
all_stopwords = set(stopwords.words('english')) # set the language
from typing import List

