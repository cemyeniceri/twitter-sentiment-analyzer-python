import twitterClient

import naive_bayes_classifier, max_entropy_classifier, libsvm_classifier


def app():

    keyword = 'apple'
    method = 'svm'
    training_required = 1
    classify_enabled = 1

    while 1:
        print("\n\n\n-------------------------- 0 --------------------------")
        print("Please select a algorithm:")
        print("0 - Exit from application")
        print("1 - Naive Bayes Algorithm")
        print("2 - Maximum Entropy Algorithm")
        print("3 - Support Vector Machine (SVM)")

        input_data = raw_input("Enter Value : ")
        if input_data.isdigit() and 0 < int(input_data) <= 3:
            method = int(input_data)
        elif input_data.isdigit() and int(input_data) == 0:
            break
        else:
            print("Wrong Selection! Read : " + input_data)
            continue

        print("\nCalculate Accuracy or Classify?")
        print("0 - Exit from application")
        print("1 - Accuracy")
        print("2 - Classify")
        input_data = raw_input("Enter Value : ")
        if input_data.isdigit() and 0 < int(input_data) <= 2:
            classify_enabled = int(input_data) - 1
        elif input_data.isdigit() and int(input_data) == 0:
            break
        else:
            print("Wrong Selection! Read : " + input_data)
            continue

        if classify_enabled == 1:
            print("\nIs training required?")
            print("0 - Exit from application")
            print("1 - Not required")
            print("2 - Required")
            input_data = raw_input("Enter Value : ")
            if input_data.isdigit() and 0 < int(input_data) <= 2:
                training_required = int(input_data) - 1
            elif input_data.isdigit() and int(input_data) == 0:
                break
            else:
                print("Wrong Selection! Read : " + input_data)
                continue

            print("\nEnter a keyword for test")
            input_data = raw_input("Enter Keyword : ")
            keyword = input_data

            print("Selected Keyword: " + keyword)

        print("-------------------------- = --------------------------\n\n\n")

        training_data_file = 'data-set/Airline-Sentiment.csv'

        if classify_enabled:
            tweets = twitterClient.getTwitterData(keyword)
            if not tweets:
                print("Tweet couldn't be fetched")
                return

        if method == 1:
            print("Method: naive bayes")
            if classify_enabled:

                classifier_dump_file = 'data-set/nb_trained_model.pickle'
                nb = naive_bayes_classifier.NaiveBayesClassifier(training_data_file, classifier_dump_file)
                nb.classify(tweets, training_required)

            else:
                classifier_dump_file = 'data-set/nb_trained_model_acc.pickle'
                nb = naive_bayes_classifier.NaiveBayesClassifier(training_data_file, classifier_dump_file)
                nb.accuracy()

        elif method == 2:
            print("Method: Max Entropy")
            if classify_enabled:

                classifier_dump_file = 'data-set/maxent_trained_model.pickle'
                maxent = max_entropy_classifier.MaxEntClassifier(training_data_file, classifier_dump_file)
                maxent.classify(tweets, training_required)

            else:
                classifier_dump_file = 'data-set/maxent_trained_model_acc.pickle'
                maxent = max_entropy_classifier.MaxEntClassifier(training_data_file, classifier_dump_file)
                maxent.accuracy()

        elif method == 3:
            print("Method: Support Vector Machine")
            if classify_enabled:

                classifier_dump_file = 'data-set/svm_trained_model.pickle'
                sc = libsvm_classifier.SVMClassifier(training_data_file, classifier_dump_file)
                sc.classify(tweets, training_required)

            else:
                classifier_dump_file = 'data-set/svm_trained_model_acc.pickle'
                sc = libsvm_classifier.SVMClassifier(training_data_file, classifier_dump_file)
                sc.accuracy()
    #end of while


if __name__ == "__main__":
    app()
