

df = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
df.replace(['No', 'Yes'], [0,1], inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
features = df.drop(columns='RainTomorrow', axis=1)
Y = df['RainTomorrow']

df = df.astype(float)
x_train, x_test, y_train, y_test = train_test_split( features,Y, test_size=0.2, random_state=10)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

LinearReg = LinearRegression()
LinearReg.fit (x_train, y_train)
print ('Coefficients: ', LinearReg.coef_)

y_pred = LinearReg.predict(x_test)

print('Coefficients:', LinearReg.coef_)
print('Intercept:', LinearReg.intercept_)

from sklearn.metrics import r2_score

LinearRegression_MAE = np.mean(np.absolute(y_pred - y_test))
LinearRegression_MSE = np.mean((y_pred -y_test)**2)
LinearRegression_R2 = r2_score(y_test, y_pred)
print("Mean absolute error: %.2f" % LinearRegression_MAE)
print("Residual sum of squares (MSE): %.2f" % LinearRegression_MSE)
print("R2-score: %.2f" % LinearRegression_R2 )
LinearRegression_RMSE = np.sqrt(LinearRegression_MSE)
print('Root Mean Squared Error:%.2f' % LinearRegression_RMSE)

dict = {'error_type':['LinearRegression_MAE','LinearRegression_MSE','LinearRegression_R2','LinearRegression_RMSE'],

        'value':[LinearRegression_MAE,LinearRegression_MSE,LinearRegression_R2,LinearRegression_RMSE]}
from tabulate import tabulate
Report = pd.DataFrame(dict)
print(tabulate(Report, headers = 'keys', tablefmt = 'psql'))

from sklearn.neighbors import KNeighborsClassifier
k = 4
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
neigh
predictions = neigh.predict(x_test)
predictions[0:5]

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,precision_score,recall_score,roc_auc_score,roc_curve
Accuracy_Score = metrics.accuracy_score(y_test, predictions)
F1_Score = metrics.f1_score(y_test, predictions)
Log_Loss = metrics.log_loss(y_test, predictions)
print(" Accuracy Score: ",Accuracy_Score)
print(" F1 score : ", F1_Score)
print(" Loss : ", Loss)
confusion_matrix = confusion_matrix(y_test,predictions)
print('Confusion Matrix:',confusion_matrix)
precision = precision_score(y_test,predictions)
print('Precision:', precision)
roc_auc = roc_auc_score(y_test,predictions)
print('ROC-AUC Score:', roc_auc)

recall = recall_score(y_test,predictions)
print('Recall:', recall)

