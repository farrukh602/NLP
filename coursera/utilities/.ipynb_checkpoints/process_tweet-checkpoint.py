import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
def process_tweet(tweet):
    # remove old style retweet text 'RT'
    tweet=re.sub(r'^RT[\s]+','',tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+','',tweet)
    # remove hastags
    tweet = re.sub(r'#','',tweet)

    # tokenization
    tokenizer = TweetTokenizer(preserve_case=False,
                              strip_handles=True,
                              reduce_len=True)
    tweet_tokens=tokenizer.tokenize(tweet)

    # removing the stop wods and punctuations
    stop_words = stopwords.words('english')
    tweet_clean = [word for word in tweet_tokens if word not in stop_words and word not in string.punctuation]

    # stemming the tweet words
    stemmer=PorterStemmer()
    stemmed_tweet = [stemmer.stem(word) for word in tweet_clean]
    
    
    return stemmed_tweet
    