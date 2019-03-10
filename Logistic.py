from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from Dataset import Dataset

ds = Dataset.generate()

y = ds[:,10]
X = ds[:,0:9]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))