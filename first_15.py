from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import numpy as np

data = pd.read_csv("data/2024_first_15.csv")
data = data.dropna()

factors = ['side', 'firstblood', 'firstdragon', 'void_grubs', 'opp_void_grubs', 'firsttower', 'turretplates', 'opp_turretplates', 'golddiffat10', 'xpdiffat10', 'killsat10', 'opp_killsat10', 'golddiffat15', 'xpdiffat15', 'killsat15', 'opp_killsat15']

x = data[factors]
y = data.result

# split the data into training data and test data
# use a 75 25 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

# fit the model using logistic regression
model = LogisticRegression(random_state=16, max_iter=4000)
model.fit(x_train, y_train)

# use the model to make predictions with the test data
y_pred = model.predict(x_test)

# create a confusion matrix of the results
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# calculate the number of correct and incorrect predictions
total_correct = cnf_matrix[0][0] + cnf_matrix[1][1]
total_wrong = cnf_matrix[0][1] + cnf_matrix[1][0]

print('Total correct predictions:', total_correct)
print('Total incorrect predictions:', total_wrong)

avg = (float(total_correct) / float(total_correct + total_wrong)) * 100
print(f"Average Correct Prediction Rate: %.2f%%" % avg)
