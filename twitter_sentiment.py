import twitterClient

import naive_bayes_classifier, max_entropy_classifier, libsvm_classifier


def app():
    keyword = 'apple'
    method = 'svm'

    training_required = 1
    classify_enabled = 1

    print(method)
    print(keyword)

    training_data_file = 'data-set/Airline-Sentiment.csv'

    if classify_enabled:
        tweets = twitterClient.getTwitterData(keyword)
        if not tweets:
            print("Tweet couldn't be fetched")
            return

    if method == 'naivebayes':
        if classify_enabled:

            classifier_dump_file = 'data-set/nb_trained_model.pickle'
            nb = naive_bayes_classifier.NaiveBayesClassifier(training_data_file, classifier_dump_file)
            nb.classify(tweets, training_required)

        else:
            classifier_dump_file = 'data-set/nb_trained_model_acc.pickle'
            nb = naive_bayes_classifier.NaiveBayesClassifier(training_data_file, classifier_dump_file)
            nb.accuracy()

    elif method == 'maxentropy':
        if classify_enabled:

            classifier_dump_file = 'data-set/maxent_trained_model.pickle'
            maxent = max_entropy_classifier.MaxEntClassifier(training_data_file, classifier_dump_file)
            maxent.classify(tweets, training_required)

        else:
            classifier_dump_file = 'data-set/maxent_trained_model_acc.pickle'
            maxent = max_entropy_classifier.MaxEntClassifier(training_data_file, classifier_dump_file)
            maxent.accuracy()

    elif method == 'svm':

        if classify_enabled:

            classifier_dump_file = 'data-set/svm_trained_model.pickle'
            sc = libsvm_classifier.SVMClassifier(training_data_file, classifier_dump_file)
            sc.classify(tweets, training_required)

        else:
            classifier_dump_file = 'data-set/svm_trained_model_acc.pickle'
            sc = libsvm_classifier.SVMClassifier(training_data_file, classifier_dump_file)
            sc.accuracy()


if __name__ == "__main__":
    app()
