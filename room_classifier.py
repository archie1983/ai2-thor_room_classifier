import pickle
import os.path
from sklearn import model_selection as cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer

from time import time
from sklearn.svm import SVC

# store our labels into a pickle file
labels_fname = "labels_shuffled.pkl"
features_fname = "features_for_each_label.pkl"

file = open(features_fname,'rb')
features_for_each_label = pickle.load(file)
file.close()

file = open(labels_fname,'rb')
labels_shuffled = pickle.load(file)
file.close()

# Now create training and testing sets
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features_for_each_label, labels_shuffled, test_size=0.1, random_state=1983)

# Now we will turn the texts into numerical vectors so that we can use that for machine learning
#vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
vectorizer = TfidfVectorizer()

features_train_vectorized = vectorizer.fit_transform(features_train)
features_test_vectorized  = vectorizer.transform(features_test)


#print "tfidf.get_stop_words(): ",tfidf.get_stop_words()
#print "vector: ",vector
print(features_train_vectorized.shape)
print(features_test_vectorized.shape)
vectorizer.get_feature_names_out()


# Finally learn and test how good model have we got
clf = SVC(kernel="rbf", C=10000.0)

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

t0 = time()
clf.fit(features_train_vectorized, labels_train)
print("training time:", round(time()-t0, 3), "s")

t0 = time()
pred = clf.predict(features_test_vectorized)
print("prediction time:", round(time()-t0, 3), "s")

from sklearn.metrics import accuracy_score

to = time()
acc = accuracy_score(pred, labels_test)
print("accuracy calculation time:", round(time()-t0, 3), "s")

print("Accuracy = ",acc)

print("Predicted Class for Elem 10:",pred[10]," Class for Elem 8:",pred[8]," Class for elem 5:", pred[5])

print("Real Class for Elem 10:",labels_test[10]," Real Class for Elem 8:",labels_test[8]," Real Class for elem 5:", labels_test[5])
#########################################################
