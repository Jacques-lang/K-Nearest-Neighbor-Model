import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
data = pandas.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

del data['SkinThickness']
data.head()
#%%
data['Outcome'].value_counts()
#%%
data_corr = data.corr()
data_corr
#%%
scaler = StandardScaler()
scaler.fit(data.drop('Outcome', axis=1))
scaled_data = scaler.transform(data.drop('Outcome', axis=1))
#%%
X = scaled_data
y = data['Outcome']
#%%
X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.4, shuffle=True)
#%%
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

model_pred = model.predict(X_test)
#%%
check1 = confusion_matrix(y_test, model_pred)
print(check1)

check2 = classification_report(y_test, model_pred)
print(check2)
#%%
test = [1,85, 66,0,26.6,0.351,31]
model_test = model.predict([test])
if model_test[0] == 0:
    print("Based on the model's prediction, the patient does not have diabetes")
else:
    print("Based on the model's prediction, the patient has diabetes")
