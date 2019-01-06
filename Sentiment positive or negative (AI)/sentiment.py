import nltk
from nltk.classify import NaiveBayesClassifier

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

pos = []
with open("./pos_tweets.txt") as f:
    for i in f:
        pos.append([format_sentence(i), 'pos'])
neg = []
with open("./neg_tweets.txt") as f:
    for i in f:
        neg.append([format_sentence(i), 'neg'])
# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
classifier = NaiveBayesClassifier.train(training)
example1 = "i love you" 
s=classifier.classify(format_sentence(example1))
if(s=="neg"):
    print("Unhappy\n")
else :
    print("Happy\n")
