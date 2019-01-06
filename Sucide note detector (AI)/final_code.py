import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#training process start
messages = pandas.read_csv('./note_collection', sep='\t', quoting=csv.QUOTE_NONE, names=["label", "message"])

#print first 10 massege
#print (messages[:10])
messages.groupby('label').describe()
messages['length'] = messages['message'].map(lambda text: len(text))

#print some massege with length,label,text
#print (messages.head())
messages.length.describe()

def split_into_tokens(message):
    #message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words

messages.message.apply(split_into_tokens)

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

messages.message.apply(split_into_lemmas)
bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
#print (len(bow_transformer.vocabulary_))
#user input massege
#message4 = messages['message'][3]
message4=""
print(message4)
#bag of word for massege
bow4=bow_transformer.transform([message4])

messages_bow = bow_transformer.transform(messages['message'])
#print 'sparse matrix shape:', messages_bow.shape
#print 'number of non-zeros:', messages_bow.nnz
#print 'sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))

tfidf_transformer = TfidfTransformer().fit(messages_bow)#fit training data
tfidf4 = tfidf_transformer.transform(bow4)#transform user massege into training data
messages_tfidf = tfidf_transformer.transform(messages_bow)
text_detector = MultinomialNB().fit(messages_tfidf, messages['label'])


print ("predicted: ", text_detector.predict(tfidf4)[0])
#print ("expected: ", messages.label[3])

all_predictions = text_detector.predict(messages_tfidf)
#print (all_predictions)

#score testing

#print ("accuracy: ", accuracy_score(messages['label'], all_predictions))
#print 'confusion matrix\n', confusion_matrix(messages['label'], all_predictions)
#print '(row=expected, col=predicted)'

