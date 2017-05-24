import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.cross_validation import train_test_split

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

if __name__ == '__main__':

	train = pd.read_csv("Reviews.csv", header=0, \
						delimiter=",", quoting=0)
	# = csv_import.sample(n=1000)
	print train.head
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
	print "Vectorizing.."
	# Create feature vectors
	vectorizer = TfidfVectorizer(min_df=5,
								 max_df = 0.8,
								 sublinear_tf=True,
								 use_idf=True)
	train_vectors = vectorizer.fit_transform(X_train)
	test_vectors = vectorizer.transform(X_test)
	'''print "Performing Classification 1"
	# Perform classification with SVM, kernel=rbf
	classifier_rbf = svm.SVC()
	t0 = time.time()
	classifier_rbf.fit(train_vectors, y_train)
	t1 = time.time()
	prediction_rbf = classifier_rbf.predict(test_vectors)
	t2 = time.time()
	time_rbf_train = t1-t0
	time_rbf_predict = t2-t1
	print "Performing Classification 2"
	# Perform classification with SVM, kernel=linear
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, y_train)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(test_vectors)
	t2 = time.time()
	time_linear_train = t1-t0
	time_linear_predict = t2-t1

	# Perform classification with SVM, kernel=linear
	
	classifier_liblinear = svm.LinearSVC()
	t0 = time.time()
	classifier_liblinear.fit(train_vectors, y_train)
	t1 = time.time()
	prediction_liblinear = classifier_liblinear.predict(test_vectors)
	t2 = time.time()
	time_liblinear_train = t1-t0
	time_liblinear_predict = t2-t1
	'''
	#twitter
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
	tweet_vectors = vectorizer.transform(clean_tweets)
	print "Performing Classification on tweets"
	# Perform classification with SVM, kernel=linear
	classifier_linear = svm.SVC(kernel='linear')
	t0 = time.time()
	classifier_linear.fit(train_vectors, y_train)
	t1 = time.time()
	prediction_linear = classifier_linear.predict(tweet_vectors)
	t2 = time.time()
	time_linear_train = t1-t0
	time_linear_predict = t2-t1
	
	# Print results in a nice table
	'''
	print("Results for SVC(kernel=rbf)")
	print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
	print(classification_report(y_test, prediction_rbf))
	'''
	print("Results for SVC(kernel=linear)")
	print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
	print(classification_report(tweets["Sentiment"], prediction_linear))
	'''
	print("Results for LinearSVC()")
	print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
	print(classification_report(y_test, prediction_liblinear))
	'''