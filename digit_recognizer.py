from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import pandas as pd
import pickle
import os

path = r'datasets/digit-recognizer'
train_path = os.path.join(path, 'train.csv')
test_path = os.path.join(path, 'test.csv')

train_pd = pd.read_csv(train_path)
test_pd = pd.read_csv(test_path)

y = train_pd['label'].values
X = train_pd.drop('label', axis=1)
X1 = test_pd.values

# Show digit
digit = X.iloc[21, :].values
digit_image = digit.reshape(28, 28)
plt.imshow(digit_image, cmap='binary')
plt.axis('off')
plt.show()

# splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Training models
knn = KNeighborsClassifier(n_neighbors=25, weights='uniform')
svm = SVC()
tree = DecisionTreeClassifier()


# print('training data')
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)
tree.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
svm_pred = svm.predict(X_test)
tree_pred = tree.predict(X_test)

# Loading trained models
# print('loading model for prediction')
# knn = pickle.load(open(r'digit_recognizer_models/knn.p', 'rb'))

# print('saving data')
pickle.dump(knn, open(r'digit_recognizer_models/knn.p', 'wb'))
pickle.dump(svm, open(r'digit_recognizer_models/svc.p', 'wb'))
pickle.dump(tree, open(r'digit_recognizer_models/tree.p', 'wb'))

print('Accuracy svc:', accuracy_score(svm_pred, y_test) * 100)
print('Accuracy knn:', accuracy_score(knn_pred, y_test) * 100)
print('Accuracy tree:', accuracy_score(tree_pred, y_test) * 100)

# Predict X1 for test_dataframe(test_df)
print('predicting')
ans = knn.predict(X1)

# print('writing to csv file')
with open('digit_answers.csv', 'w') as file:
    file.write('ImageId,Label\n')
    for index, digit in enumerate(ans):
        file.write(f'{index + 1},{digit}\n')

print('done')
