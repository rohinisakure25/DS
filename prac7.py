import pandas as pd
import math

D1 = 'I love data science'
D2 = 'I love machine learning'

# Tokenization
bagA = D1.lower().split()
bagB = D2.lower().split()
uniqueWords = set(bagA).union(set(bagB))

# Term Counts
dictA = dict.fromkeys(uniqueWords, 0)
for word in bagA: dictA[word] += 1
dictB = dict.fromkeys(uniqueWords, 0)
for word in bagB: dictB[word] += 1

# TF Calculation
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    count = len(bagOfWords)
    for word, val in wordDict.items():
        tfDict[word] = val / float(count)
    return tfDict

# IDF Calculation
def computeIDF(docList):
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0: idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

# TF-IDF Calculation
def computeTFIDF(tfBag, idfs):
    tfidf = {}
    for word, val in tfBag.items():
        tfidf[word] = val * idfs[word]
    return tfidf

# Execution
tfA = computeTF(dictA, bagA)
tfB = computeTF(dictB, bagB)
idfs = computeIDF([dictA, dictB])
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)

# Resulting Matrix
df = pd.DataFrame([tfidfA, tfidfB], index=['D1', 'D2'])
print("TF-IDF Matrix:")
print(df)

import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Step 1: Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Step 2 & 3: Tokenization
text = "Tokenization is the first step in text analytics. The process of breaking text into paragraphs, sentences, or words."
print("--- Sentence Tokenization ---")
print(sent_tokenize(text))

print("\n--- Word Tokenization ---")
tokens = word_tokenize(text)
print(tokens)

# Step 4: Stop Words Removal & Cleaning
stop_words = set(stopwords.words("english"))
clean_text = re.sub('[^a-zA-Z]', ' ', text)
word_tokens = word_tokenize(clean_text.lower())
filtered_text = [w for w in word_tokens if w not in stop_words]

print("\n--- Filtered Text (No Stop Words) ---")
print(filtered_text)

# Step 5 & 6: Stemming
ps = PorterStemmer()
words_to_stem = ["wait", "waiting", "waited", "waits"]
stemmed_words = [ps.stem(w) for w in words_to_stem]

print("\n--- Stemming Results ---")
print(dict(zip(words_to_stem, stemmed_words)))

# Step 7: POS Tagging
data = "The pink sweater fit her perfectly"
pos_tokens = word_tokenize(data)
print("\n--- POS Tagging ---")
print(nltk.pos_tag(pos_tokens))
