from string import punctuation
import nltk
import pandas as pd
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords

doc1 = open("CELEX_32016R0679_EN_TXT.txt", "r")
doc1Txt = doc1.read()
txt = ''.join(c for c in doc1Txt if not c.isdigit())
txt = ''.join(c for c in txt if c not in punctuation).lower()
txt = ' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])
words = nltk.tokenize.word_tokenize(txt)
fdist = FreqDist(words)
count_frame = pd.DataFrame(fdist, index =[0]).T
count_frame.columns = ['Count']
counts = count_frame.sort_values('Count', ascending = False)
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
counts['Count'][:60].plot(kind = 'bar', ax = ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of the word')
ax.set_xlabel('Word')
plt.show()

#To get insights about the info...
print ('Summary:')
#Summarize a function of gensim
print (summarize(doc1Txt, ratio=0.00125))
print ('\nKeywords:')
#Keywords a function of gensim
print (keywords(doc1Txt, ratio=0.005))
