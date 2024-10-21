import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

nltk.download('gutenberg')
nltk.download('punkt_tab')
nltk.download('stopwords')

macbeth_text = nltk.corpus.gutenberg.raw('shakespeare-macbeth.txt')

tokens = word_tokenize(macbeth_text)

total_words = len(tokens)
print(f"Total number of words in the text: {total_words}")

fdist_before = FreqDist([word.lower() for word in tokens if word.isalpha()])

most_common_before = fdist_before.most_common(10)
print(f"Most common words before stopword removal: {most_common_before}")

words_before = [word for word, _ in most_common_before]
counts_before = [count for _, count in most_common_before]

plt.figure(figsize=(10, 6))
plt.bar(words_before, counts_before, color='blue')
plt.title('Top 10 most common words before stopword removal')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

stop_words = set(stopwords.words('english'))
tokens_clean = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]

fdist_after = FreqDist(tokens_clean)

most_common_after = fdist_after.most_common(10)
print(f"Most common words after stopword and punctuation removal: {most_common_after}")

words_after = [word for word, _ in most_common_after]
counts_after = [count for _, count in most_common_after]

plt.figure(figsize=(10, 6))
plt.bar(words_after, counts_after, color='green')
plt.title('Top 10 most common words after stopword removal')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()
