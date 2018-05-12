import csv
import pickle
import os.path
import nltk.classify

import classifier_helper


# start class
class NaiveBayesClassifier:
    """ Naive Bayes Classifier """

    # variables
    # start __init__
    def __init__(self, training_data_file, classifier_dump_file):
        # Instantiate classifier helper
        self.helper = classifier_helper.ClassifierHelper()
        self.training_data_file = training_data_file
        self.classifier_dump_file = classifier_dump_file
        self.test_tweet_items = []

    # end

    # start getUniqData
    def getTweetsWithUniqueWords(self, tweets):
        tweets_with_unique_words = []
        for tweet in tweets:
            word_set = []
            words = tweet.split()
            for word in words:
                if word not in word_set and self.helper.is_ascii(word):
                    word_set.append(word)
            # end inner loop
            tweet_str = " ".join(str(x) for x in word_set)
            tweets_with_unique_words.append(tweet_str)
            # end outer loop
        return tweets_with_unique_words

    # end

    # start getProcessedTweets
    def getFeatureVectorList(self, tweets):
        feature_vector = []
        for tweet in tweets:
            feature_vector.append(self.helper.getFeatureVector(tweet))
        return feature_vector

    # end

    # start getProcessedTweets
    def getProcessedTweetList(self, tweets):
        processed_tweets = []
        for tweet in tweets:
            processed_tweets.append(self.helper.process_tweet(tweet))
            # end loop
        return processed_tweets

    # end

    # start getNBTrainedClassifier
    def getNBTrainedClassifer(self, training_data_file, classifier_dump_file):
        # read all tweets and labels
        tweet_items = self.getFilteredTrainingData(training_data_file)

        tweets = []
        feature_list = []
        for (tweet, sentiment) in tweet_items:
            processed_tweet = self.helper.process_tweet(tweet)
            feature_vector = self.helper.getFeatureVector(processed_tweet)
            feature_list.extend(feature_vector)
            tweets.append((feature_vector, sentiment))
        # end loop

        # Remove feature_list duplicates
        self.helper.featureList = list(set(feature_list))
        training_set = nltk.classify.apply_features(self.helper.extract_features, tweets)
        # Write back classifier and word features to a file
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        outfile = open(classifier_dump_file, 'wb')
        pickle.dump(classifier, outfile)
        outfile.close()
        return classifier

    # end

    # start getFilteredTrainingData
    def getFilteredTrainingData(self, training_data_file):
        fp = open(training_data_file, 'rb')
        min_count = self.getMinCount(training_data_file)
        training_data_count = int(min_count * 0.75)
        neg_count, pos_count, neut_count = 0, 0, 0

        reader = csv.reader(fp, delimiter=',', quotechar='"', escapechar='\\')
        tweet_items = []
        for row in reader:
            processed_tweet = self.helper.process_tweet(row[1])
            sentiment = row[0]

            if sentiment == 'neutral':
                if neut_count < training_data_count:
                    tweet_item = processed_tweet, sentiment
                    tweet_items.append(tweet_item)
                elif neut_count < min_count:
                    tweet_item = processed_tweet, sentiment
                    self.test_tweet_items.append(tweet_item)
                neut_count += 1
            elif sentiment == 'positive':
                if pos_count < training_data_count:
                    tweet_item = processed_tweet, sentiment
                    tweet_items.append(tweet_item)
                elif pos_count < min_count:
                    tweet_item = processed_tweet, sentiment
                    self.test_tweet_items.append(tweet_item)
                pos_count += 1
            elif sentiment == 'negative':
                if neg_count < training_data_count:
                    tweet_item = processed_tweet, sentiment
                    tweet_items.append(tweet_item)
                elif neg_count < min_count:
                    tweet_item = processed_tweet, sentiment
                    self.test_tweet_items.append(tweet_item)
                neg_count += 1
        # end loop
        return tweet_items

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

    # start classify
    def classify(self, tweets_fetched, training_required):

        # call training model
        if training_required:
            classifier = self.getNBTrainedClassifer(self.training_data_file, self.classifier_dump_file)
        else:
            if os.path.isfile(self.classifier_dump_file):
                classifier_file = open(self.classifier_dump_file)
                classifier = pickle.load(classifier_file)
                classifier_file.close()
            else:
                classifier = self.getNBTrainedClassifer(self.training_data_file, self.classifier_dump_file)

        orig_tweets = self.getTweetsWithUniqueWords(tweets_fetched)
        processed_tweets = self.getProcessedTweetList(orig_tweets)
        feature_vectors = self.getFeatureVectorList(processed_tweets)

        count = 0
        results = {}
        for feature_vector in feature_vectors:
            label = classifier.classify(self.helper.extract_features(feature_vector))
            result = {'tweet': orig_tweets[count], 'label': label}
            results[count] = result
            count += 1
        print(results)

    # end

    # start accuracy
    def accuracy(self):
        classifier_acc = self.getNBTrainedClassifer(self.training_data_file, self.classifier_dump_file)
        total = 0
        wrong = 0

        corrects = {'positive': 0, 'negative': 0, 'neutral': 0}

        for (tweet, sentiment) in self.test_tweet_items:
            processed_tweet = self.helper.process_tweet(tweet)
            test_feature_vector = self.helper.getFeatureVector(processed_tweet)
            label = classifier_acc.classify(self.helper.extract_features(test_feature_vector))
            if label == sentiment:
                corrects[sentiment] += 1
            else:
                wrong += 1
            total += 1
        # end loop
        correct = corrects['positive'] + corrects['negative'] + corrects['neutral']
        print(corrects)

        accuracy = (float(correct) / total) * 100
        print('Total = %d, Correct = %d, Wrong = %d, Accuracy = %.2f' % \
              (total, correct, wrong, accuracy))
        # end

# end class
