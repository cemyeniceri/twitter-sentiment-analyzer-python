from svmutil import *
import csv
import classifier_helper


# start class
class SVMClassifier:
    """ SVM Classifier """

    # variables
    # start __init__
    def __init__(self, training_data_file, classifier_dump_file):

        # Instantiate classifier helper
        self.helper = classifier_helper.ClassifierHelper()
        self.training_data_file = training_data_file
        self.classifier_dump_file = classifier_dump_file
        self.test_tweet_items = []

    # end

    # start getUniqueData
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
    def getProcessedTweetList(self, tweets):
        processed_tweets = []
        for tweet in tweets:
            processed_tweets.append(self.helper.process_tweet(tweet))
            # end loop
        return processed_tweets

    # end

    # start getNBTrainedClassifier
    def getSVMTrainedClassifer(self, training_data_file, classifier_dump_file):
        # read all tweets and labels
        tweet_items = self.getFilteredTrainingData(training_data_file)

        tweets = []
        feature_list = []
        for (tweet, sentiment) in tweet_items:
            tweets.append((tweet, sentiment))
            feature_vector = self.helper.getFeatureVector(tweet)
            feature_list.extend(feature_vector)
        # end loop

        # Remove feature_list duplicates
        self.helper.featureList = list(set(feature_list))
        feature_vector_label = self.helper.getSVMFeatureVectorAndLabels(tweets)

        feature_vectors = feature_vector_label['feature_vector']
        labels = feature_vector_label['labels']

        # SVM Trainer
        problem = svm_problem(labels, feature_vectors)
        # '-q' option suppress console output
        param = svm_parameter('-q')
        param.kernel_type = LINEAR
        # param.show()
        classifier = svm_train(problem, param)
        svm_save_model(classifier_dump_file, classifier)
        return classifier

    # end

    # start getFilteredTrainingData
    def getFilteredTrainingData(self, training_data_file):
        fp = open(training_data_file, 'rb')
        # min_count = self.getMinCount(training_data_file)
        min_count = 100
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
            classifier = self.getSVMTrainedClassifer(self.training_data_file, self.classifier_dump_file)
        else:
            if os.path.isfile(self.classifier_dump_file):
                classifier = svm_load_model(self.classifier_dump_file)
            else:
                classifier = self.getSVMTrainedClassifer(self.training_data_file, self.classifier_dump_file)

        orig_tweets = self.getTweetsWithUniqueWords(tweets_fetched)
        processed_tweets = self.getProcessedTweetList(orig_tweets)

        test_feature_vector = self.helper.getSVMFeatureVector(processed_tweets)
        p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector), \
                                               test_feature_vector, classifier)
        count = 0
        results = {}
        for t in processed_tweets:
            label = p_labels[count]
            if label == 0:
                label = 'positive'
            elif label == 1:
                label = 'negative'
            elif label == 2:
                label = 'neutral'
            result = {'tweet': t, 'label': label}
            results[count] = result
            count += 1
        print(results)
        # end loop

    # end

    # start accuracy
    def accuracy(self):

        count = 0
        total, correct, wrong = 0, 0, 0

        classifier_acc = self.getSVMTrainedClassifer(self.training_data_file, self.classifier_dump_file)

        test_tweets = []
        for (t, l) in self.test_tweet_items:
            test_tweets.append(t)

        test_feature_vector = self.helper.getSVMFeatureVector(test_tweets)

        p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector), \
                                               test_feature_vector, classifier_acc)

        for (t, l) in self.test_tweet_items:
            label = p_labels[count]
            if label == 0:
                label = 'positive'
            elif label == 1:
                label = 'negative'
            elif label == 2:
                label = 'neutral'

            if label == l:
                correct += 1
            else:
                wrong += 1
            total += 1
            count += 1
        # end loop
        self.accuracy = (float(correct) / total) * 100
        print('Total = %d, Correct = %d, Wrong = %d, Accuracy = %.2f' % \
              (total, correct, wrong, self.accuracy))
    # end

# end class
