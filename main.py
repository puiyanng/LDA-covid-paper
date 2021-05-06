import preprocessing, gensim
from gensim import corpora
import pickle
import pandas as pd
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

text_data = []
source = pd.read_csv('data/clean_comm_use.csv')
noncomm_source = pd.read_csv('data/clean_comm_use.csv')
papers = source.text

for paper in tqdm(papers):
    tokens = preprocessing.prepare_text_for_lda(paper)
    text_data.append(tokens)

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

pickle.dump(corpus, open('corpus_comm_use.pkl', 'wb'))
dictionary.save('dictionary_comm_use.gensim')

NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model_comm_use.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)

#Evaluation Part
print('Perplexity: ', ldamodel.log_perplexity(corpus))

for t in range(ldamodel.num_topics):
    ldamodel.show_topic(t, 200)

#Visualization
for t in range(ldamodel.num_topics):
    plt.figure()
    plt.imshow(WordCloud().fit_words(dict(ldamodel.show_topic(t, 200))))
    plt.axis("off")
    plt.title("Topic #" + str(t))
    plt.show()
