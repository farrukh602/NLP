import numpy as np  # Ensure numpy is imported

def process_tweet(tweet):
    import re
    import string
    
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import TweetTokenizer
    
    # remove old style retweet text 'RT'
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    tweet = re.sub(r'#', '', tweet)

    # tokenization
    tokenizer = TweetTokenizer(preserve_case=False,
                               strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    # removing the stop words and punctuations
    stop_words = stopwords.words('english')
    tweet_clean = [word for word in tweet_tokens if word not in stop_words and word not in string.punctuation]

    # stemming the tweet words
    stemmer = PorterStemmer()
    stemmed_tweet = [stemmer.stem(word) for word in tweet_clean]
    
    return stemmed_tweet

def count_frequency(tweets, labels):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        labels: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        frequencies: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    label_list = np.squeeze(labels).tolist()
    freqs = {}
    for label, tweet in zip(label_list, tweets):
        for word in process_tweet(tweet):
            pair = (word, label)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1
    return freqs
