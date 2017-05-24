from os import path
from wordcloud import WordCloud
import pandas as pd
import re
def sub_punctuation(text):

	subbed_text = re.sub("[^a-zA-Z]",           # The pattern to search for
						  " ",                   # The pattern to replace it with
						  text).lower()
	return subbed_text
train = pd.read_csv("Reviews.csv", header=0, \
						delimiter=",", quoting=0)
text = " ".join(sub_punctuation(w) for w in train["Text"])
wordlist = str.split(text)
print wordlist[:1000]
print "Length: " + str(len(wordlist))
#d = path.dirname(__file__)

# Read the whole text.
#text = open(path.join(d, 'constitution.txt')).read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# The pil way (if you don't have matplotlib)
# image = wordcloud.to_image()
# image.show()
