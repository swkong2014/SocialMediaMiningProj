import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

def main():
	train = pd.read_csv("Reviews.csv", header=0, \
						delimiter=",", quoting=0)
	print train.shape
	print train.columns.values
	#clean_review = review_to_words( train["Text"][0] )
	#print clean_review
	# Get the number of reviews based on the dataframe column size
	num_reviews = train["Text"].size

	# Initialize an empty list to hold the clean reviews
	clean_train_reviews = []
	# Loop over each review; create an index i that goes from 0 to the length
	# of the movie review list 
	for i in xrange( 0, num_reviews ):
		# Call our function for each one, and add the result to the list of
		# clean reviews
		if( (i+1)%1000 == 0 ):
			print "Review %d of %d\n" % ( i+1, num_reviews ) 
		clean_train_reviews.append( review_to_words( train["Text"][i] ))#,\
									#train["Sentiment"][i]] )
	X_train, X_test, y_train, y_test = train_test_split(clean_train_reviews, train["Sentiment"].values, test_size=0.2, random_state=42)
	
	print "Creating the bag of words...\n"
	vectorizer = CountVectorizer(analyzer = "word",   \
							 tokenizer = None,    \
							 preprocessor = None, \
							 stop_words = None,   \
							 max_features = 5000)
	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(X_train)
	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	train_data_features = train_data_features.toarray()
	#vocab = vectorizer.get_feature_names()
	#print vocab
	#dist = np.sum(train_data_features, axis=0)
	# For each, print the vocabulary word and the number of times it 
	# appears in the training set
	#for tag, count in zip(vocab, dist):
	#	print count, tag
	print "Training the random forest..."
	# Initialize a Random Forest classifier with 100 trees
	forest = RandomForestClassifier(n_estimators = 10) 

	# Fit the forest to the training set, using the bag of words as 
	# features and the sentiment labels as the response variable
	#
	# This may take a few minutes to run
	forest = forest.fit( train_data_features, y_train )
	# Read the test data
	#test = pd.read_csv("testData.csv", header=0, delimiter=",", \
	#				   quoting=0 )

	# Create an empty list and append the clean reviews one by one
	#num_reviews = len(test["Text"])
	#clean_test_reviews = [] 
	#print "Cleaning and parsing the test set movie reviews...\n"
	#for i in xrange(0,num_reviews):
	#	if( (i+1) % 1000 == 0 ):
	#		print "Review %d of %d\n" % (i+1, num_reviews)
	#	clean_review = review_to_words( test["Text"][i] )
	#	clean_test_reviews.append( clean_review )

	# Get a bag of words for the test set, and convert to a numpy array
	test_data_features = vectorizer.transform(X_test)
	test_data_features = test_data_features.toarray()

	# Use the random forest to make sentiment label predictions
	result = forest.predict(test_data_features)
	result_train = forest.predict(train_data_features)
	# Copy the results to a pandas dataframe with an "id" column and
	# a "sentiment" column
	output = pd.DataFrame( data={"Sentiment":result} )

	# Use pandas to write the comma-separated output file
	output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=0 )
	
	score = metrics.f1_score(y_test, result)
	score_train = metrics.f1_score(y_train, result_train)
	
	pscore = metrics.accuracy_score(y_test, result)
	pscore_train = metrics.accuracy_score(y_train, result_train)

	tweets = pd.read_csv("tweets-labelled.csv", header=0, \
						quoting=0)
	tweet_count = len(tweets)
	clean_tweets = []
	for i in xrange( 0, tweet_count):
		# Call our function for each one, and add the result to the list of
		# clean reviews
		if( (i+1)%1000 == 0 ):
			print "Tweet %d of %d\n" % ( i+1, tweet_count ) 
		clean_tweets.append( review_to_words( tweets["Text"][i] ))
	tweet_data_features= vectorizer.transform(clean_tweets).toarray()
	tweet_result = forest.predict(tweet_data_features)
	tweet_score = metrics.f1_score(tweets["Sentiment"], tweet_result)
	tweet_acc = metrics.accuracy_score(tweets["Sentiment"], tweet_result)
	
	print "Train score: " + str(score_train)
	print "Test score: " + str(score)
	print "Train Accuracy: " + str(pscore_train)
	print "Test Accuracy: " + str(pscore)
	print "Tweet score: " + str(tweet_score)
	print "Tweet Accuracy: " + str(tweet_acc)


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
	
if __name__ == "__main__":
    main()
else:
	print "imported"
	

