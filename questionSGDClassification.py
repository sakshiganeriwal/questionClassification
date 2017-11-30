"""Simple Question Classifier using TF-IDF or Bag of Words Model"""

import sys
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn import metrics


if __name__ == "__main__":

    # the training data folder must be passed as first argument
    training_data = sys.argv[1]
    dataset = load_files(training_data, shuffle=False)
    # print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (1, 3)), 
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
    }

    gridSearch = GridSearchCV(pipeline, parameters, n_jobs=-1)
    gridSearch.fit(docs_train, y_train)

    print(gridSearch.grid_scores_)

    #Predict the outcome on the testing set and store it in a variable named y_predicted
    y_predicted = gridSearch.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))

    #confusion matrix
    confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
    print(confusionMatrix)

    import matplotlib.pyplot as plt
    plt.matshow(confusionMatrix)
    plt.show()
