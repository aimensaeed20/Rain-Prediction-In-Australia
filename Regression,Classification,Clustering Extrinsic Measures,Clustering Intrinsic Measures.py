

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

from sklearn.metrics import mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_rand_score
mi = mutual_info_score(y_test, predictions)

print('Mutual Information:', mi)
ri = adjusted_rand_score(y_test,predictions)

print('Rand Index:', ri)
ari = adjusted_rand_score(y_test,predictions)
print('Adjusted Rand Index:', ari)

from sklearn.datasets import make_blobs

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)


for n_clusters in range(2, 6):
    kmeans = KMeans(n_clusters=3, random_state=0)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)

    print(f"For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

  
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
    plt.title(f'KMeans Clustering (n_clusters={n_clusters})')
    plt.show()

from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=0)
cluster_labels = kmeans.fit_predict(X)


db_index = davies_bouldin_score(X, cluster_labels)
print('Davies-Bouldin Index:', db_index)
