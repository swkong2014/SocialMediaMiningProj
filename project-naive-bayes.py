import nltk
from nltk.corpus import movie_reviews
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import random, string
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.tokenize import word_tokenize

stop = stopwords.words('english')

# for i in stop: print i
# print movie_reviews.raw('pos/cv957_8737.txt')
train = pd.read_csv("Reviews.csv", header=0, \
						delimiter=",", quoting=0)

def review_to_words( raw_review ):
	review_text = BeautifulSoup(raw_review).get_text()
	# Print the raw review and then the output of get_text(), for 
	# comparison
	# Use regular expressions to do a find-and-replace
	letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
						  " ",                   # The pattern to replace it with
						  review_text)  # The text to search
	lower_case = letters_only.lower()        # Convert to lower case
	words = lower_case.split()               # Split into words
	stops = set(stopwords.words("english"))
	# Remove stop words from "words"
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words )) 						
						
test_doc = [(review_to_words(row["Text"]),row["Sentiment"]) for index, row in train.sample(n=20000, random_state=42).iterrows()]

'''documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
			'''
random.shuffle(test_doc)

# all_words = FreqDist(w.lower() for w in movie_reviews.words()) # Book version
print "Generating frequenccy dist.."
p = ""
num_item = len(test_doc)
i = 0
for item in test_doc:
	if( (i+1) % 1000 == 0 ):
		print "Review %d of %d\n" % (i+1, num_item)
	p += item[0]
	i+=1
print "Tokenizing for FreqDist"
all_words = FreqDist(word_tokenize(p))
#FreqDist(word_tokenize(item[0]) for item in test_doc)
# Limit the number of features that the classifier needs to process to the 2,000 most frequent words
# word_features = all_words.keys()[:2000] # wrong
# word_features = list(all_words)[:2000]  # wrong
print "Getting most common word features.."
word_features = [w for (w, c) in all_words.most_common(3000)]
#print word_features


def document_features(document): # a feature extractor, input is a list of words in a document
    # checking whether a word occurs in a set is much faster than checking whether it occurs in a list
    document_words = set(document)
    features = {}
    for word in word_features:
		if len(word)>1:
			features['contains({})'.format(word)] = (word in document_words)
    return features #return ['contains(whatever)']

	
# print document_features(movie_reviews.words('pos/cv957_8737.txt'))
# Training and testing a classifier for document classification.
print "Generating feature set..."
featuresets = [(document_features(d), c) for (d,c) in test_doc]
train_set, test_set = featuresets[4000:], featuresets[:4000]
print "Training classifier.."
classifier = nltk.NaiveBayesClassifier.train(train_set)
# classifier = nltk.DecisionTreeClassifier.train(train_set)

# Tweets

tweets = pd.read_csv("tweets-labelled.csv", header=0, \
						delimiter=",", quoting=0)
tweet_test = [(document_features(review_to_words(row["Text"])),row["Sentiment"]) for index, row in tweets.iterrows()]
tweet_featuresets = [(document_features(d), c) for (d,c) in tweet_test]

print 

print "Test set accuracy:"
print nltk.classify.accuracy(classifier, test_set)
print "Tweet accuracy:"
print nltk.classify.accuracy(classifier, tweet_featuresets)
# print classifier.pretty_format(depth=5) # for Decision Trees
#classifier.show_most_informative_features(30) # defalut 10 
