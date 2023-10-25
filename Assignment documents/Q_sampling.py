# Importing the libraries
import pandas as pd

# Importing the datas
dataset_q = pd.read_csv('500_questions.tsv', delimiter='\t', quoting=3, encoding="ISO-8859-1")
dataset_t = pd.read_csv('topic_dataset.csv')

# nltk
import re

'''nltk.download('stopwords')
           if upto date'''
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# cleaning question dataset
corpus_q = []
for i in range(0, 500):
    question = re.sub('[^a-zA-Z]', ' ', dataset_q['question'][i])
    question = question.lower()
    question = question.split()
    ps = PorterStemmer()
    question = [ps.stem(word) for word in question if not word in set(stopwords.words('english'))]
    question = ' '.join(question)
    corpus_q.append(question)

corpus_t = []
for i in range(0, 19):
    topic = re.sub('[^a-zA-Z]', ' ', dataset_t['topic'][i])
    topic = topic.lower()
    ps = PorterStemmer()
    topic = [ps.stem(word) for word in topic if not word in set(stopwords.words('english'))]
    corpus_t.append(topic)

for i in corpus_t:
    for j in corpus_q:
        if corpus_t in corpus_q:
            list = map(corpus_t, corpus_q)
