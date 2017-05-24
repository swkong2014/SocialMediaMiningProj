import twitter
import json
import csv, sys, re
from afinn import Afinn
CONSUMER_KEY = ''
CONSUMER_SECRET = ''
OAUTH_TOKEN = ''
OAUTH_TOKEN_SECRET = ''

auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,
                           CONSUMER_KEY, CONSUMER_SECRET)

twitter_api = twitter.Twitter(auth=auth)

q = '"food"' 

count = 200

# See https://dev.twitter.com/docs/api/1.1/get/search/tweets

search_results = twitter_api.search.tweets(q=q, count=count)

outfile = "tweets.csv"
csvfile = file(outfile, "wb")
csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
print csvwriter.writerow(["Sentiment", "Text"])
afinn = Afinn()

for result in search_results["statuses"]:
	tweet = result["text"].encode('unicode_escape')
	score = afinn.score(tweet)
	if score > 0:
		score = 1
	elif score < 0:
		score = 0
	else:
		score = 2
	re.sub(r"http\S+", "", tweet.replace('\n', ' '))

	print csvwriter.writerow([score, re.sub(r"http\S+", "", result["text"].encode('unicode_escape').replace('\n', ' '))])


'''
for _ in range(5):
    print "Length of statuses", len(statuses)
    try:
        next_results = search_results['search_metadata']['next_results']
    except KeyError, e: # No more results when next_results doesn't exist
        break
        
    # Create a dictionary from next_results, which has the following form:
    # ?max_id=313519052523986943&q=NCAA&include_entities=1
    kwargs = dict([ kv.split('=') for kv in next_results[1:].split("&") ])
    
    search_results = twitter_api.search.tweets(**kwargs)
    statuses += search_results['statuses']
# Show one sample search result by slicing the list...
print json.dumps(statuses[0], indent=1)
'''