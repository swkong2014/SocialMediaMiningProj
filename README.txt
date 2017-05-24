> The project is developed on python 2.7 with Windows by Songwei Kong

> DEPENDENCIES
NLTK & scikit-learn, pandas and numpy (all included within Anaconda package)
Beautiful Soup: A Python library for pulling data out of HTML and XML files 
Twitter API: A Python library for interfacing to the Twitter REST and streaming APIs
AFINN: A wordlist-based sentiment analysis library in Python
Wordcloud: A python word cloud generator

> The following projects train the respective models and output an accuracy on command prompt:
project-naive-bayes.py
project-rand-forest.py
project-svm.py
To run the project, open command prompt, navigate to the target folder and type "python project-{name}.py"

> word-cloud.py
This script does a wordcount and creates a wordcloud for reviews.csv

> twitter-search.py
This script uses Twitter api to search for 200 tweets about "food" and performs sentiment analysis using AFINN library and is saved into a tweets.csv file

>The follow csv files are required for the project to run:
tweets-labelled.csv: a hand-labelled csv files containing target tweets for prediction
Reviews.csv: 100,000 amazon review dataset
