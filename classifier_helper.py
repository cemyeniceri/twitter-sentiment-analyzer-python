import csv
import re


class ClassifierHelper:
    # start __init__
    def __init__(self):
        self.stopWords = self.getStopWordList('data-set/stopwords.txt')
        self.featureList = []
    # end

    # start getUniqueData
    def getTweetsWithUniqueWords(self, tweets):
        tweets_with_unique_words = []
        for tweet in tweets:
            word_set = []
            words = tweet.split()
            for word in words:
                if word not in word_set and self.is_ascii(word):
                    word_set.append(word)
            # end inner loop
            tweet_str = " ".join(str(x) for x in word_set)
            tweets_with_unique_words.append(tweet_str)
            # end outer loop
        return tweets_with_unique_words

    # end

    # start getProcessedTweets
    def getProcessedTweetList(self, tweets):
        processed_tweets = []
        for tweet in tweets:
            processed_tweets.append(self.process_tweet(tweet))
            # end loop
        return processed_tweets

    # end

    # start getfeatureVector
    def getFeatureVector(self, tweet):
        feature_vector = []
        words = tweet.split()
        for word in words:
            # replace two or more with two occurrences
            word = self.replaceTwoOrMore(word)
            # strip punctuation
            word = word.strip('\'"?,.!')
            # check if it consists of only words
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*[a-zA-Z]+[a-zA-Z0-9]*$", word)
            # ignore if it is a stopWord
            if word in self.stopWords or val is None:
                continue
            else:
                feature_vector.append(word)
        return feature_vector

    # end

    # start getProcessedTweets
    def getFeatureVectorList(self, tweets):
        feature_vector = []
        for tweet in tweets:
            feature_vector.append(self.getFeatureVector(tweet))
        return feature_vector

    # end

    # start getStopWordList
    def getStopWordList(self, stop_word_list_file_name):
        # read the stopwords file and build a list
        stop_words = ['AT_USER', 'URL', 'rt', 'RT']
        fp = open(stop_word_list_file_name, 'r')
        line = fp.readline()
        while line:
            word = line.strip()
            stop_words.append(word)
            line = fp.readline()
        fp.close()
        return stop_words

    # end

    # start extract_features
    def extract_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.featureList:
            word = self.replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            features['contains(%s)' % word] = (word in document_words)
        return features

    # end

    # start replaceTwoOrMore
    def replaceTwoOrMore(self, s):
        # pattern to look for three or more repetitions of any character, including
        # newlines.
        pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
        return pattern.sub(r"\1\1", s)

    # end

    def getSVMFeatureVectorAndLabels(self, tweets):
        sorted_features = sorted(self.featureList)
        feature_vector = []
        labels = []
        for t in tweets:
            label = 0
            map = {}
            # Initialize empty map
            for w in sorted_features:
                map[w] = 0

            tweet_words = t[0]
            tweet_opinion = t[1]
            # Fill the map
            words = tweet_words.split()
            for word in words:
                if word in self.stopWords:
                    continue
                word = self.replaceTwoOrMore(word)
                word = word.strip('\'"?,.')
                if word in map:
                    map[word] = 1
            # end for loop
            values = map.values()
            feature_vector.append(values)
            if tweet_opinion == 'positive':
                label = 0
            elif tweet_opinion == 'negative':
                label = 1
            elif tweet_opinion == 'neutral':
                label = 2
            labels.append(label)
        return {'feature_vector': feature_vector, 'labels': labels}

    # end

    # start getSVMFeatureVector
    def getSVMFeatureVector(self, tweets):
        sorted_features = sorted(self.featureList)
        feature_vector = []
        for t in tweets:
            map = {}
            # Initialize empty map
            for w in sorted_features:
                map[w] = 0
            # Fill the map
            for word in t.split():
                if word in self.stopWords:
                    continue
                if word in map:
                    map[word] = 1
            # end for loop
            values = map.values()
            feature_vector.append(values)
        return feature_vector

    # end

    # start process_tweet
    def process_tweet(self, tweet):
        # Conver to lower case
        tweet = tweet.lower()
        # Convert https?://* to URL
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)
        # Convert @username to AT_USER
        tweet = re.sub('@[^\s]+', 'AT_USER', tweet)
        # Remove additional white spaces
        tweet = re.sub('[\s]+', ' ', tweet)
        # Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        # trim
        tweet = tweet.strip()
        # remove first/last " or 'at string end
        tweet = tweet.rstrip('\'"')
        tweet = tweet.lstrip('\'"')
        return tweet

    # end

    # start is_ascii
    def is_ascii(self, word):
        return all(ord(c) < 128 for c in word)
    # end

    # start getMinCount
    def getMinCount(self, training_data_file):
        fp = open(training_data_file, 'rb')
        reader = csv.reader(fp, delimiter=',', quotechar='"', escapechar='\\')
        neg_count, pos_count, neut_count = 0, 0, 0
        for row in reader:
            sentiment = row[0]
            if sentiment == 'neutral':
                neut_count += 1
            elif sentiment == 'positive':
                pos_count += 1
            elif sentiment == 'negative':
                neg_count += 1
        # end loop
        return min(neg_count, pos_count, neut_count)

    # end

    #start printResult
    def printResults(self, results):
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        for key in results:
            print(results[key])
            if results[key]['label'] == 'positive':
                positive_count += 1
            elif results[key]['label'] == 'negative':
                negative_count += 1
            elif results[key]['label'] == 'neutral':
                neutral_count += 1

        print("Positive : " + str(positive_count) + " || Negative : " + str(negative_count) + " || Neutral : " + str(neutral_count))
    #end

# end class
