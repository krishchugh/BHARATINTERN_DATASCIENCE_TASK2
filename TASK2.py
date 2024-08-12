import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(train_data.head())
print(test_data.head())
print(train_data.shape, test_data.shape)
print(train_data.info())
print(test_data.info())

print(train_data.isnull().sum())
print(test_data.isnull().sum())

def plot_distribution(column_name):
    survived = train_data[train_data['Survived'] == 1][column_name].value_counts()
    not_survived = train_data[train_data['Survived'] == 0][column_name].value_counts()
    pd.DataFrame({'Survived': survived, 'Not Survived': not_survived}).plot(kind='bar', figsize=(10, 5))

plot_distribution('Sex')
plot_distribution('Pclass')

for dataset in [train_data, test_data]:
    dataset['Title'] = dataset['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())

title_mapping = {
    'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Dr': 3, 'Rev': 3, 'Col': 3, 'Major': 3,
    'Mlle': 3, 'Countess': 3, 'Ms': 3, 'Lady': 3, 'Jonkheer': 3, 'Don': 3, 'Dona': 3, 'Mme': 3,
    'Capt': 3, 'Sir': 3
}    

for dataset in [train_data, test_data]:
    dataset['Title'] = dataset['Title'].map(title_mapping)

plot_distribution('Title')

train_data.drop(columns=['Name'], inplace=True)
test_data.drop(columns=['Name'], inplace=True)

train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

plot_distribution('Sex')

train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'), inplace=True)
test_data['Age'].fillna(test_data.groupby('Title')['Age'].transform('median'), inplace=True)

sns.kdeplot(data=train_data, x='Age', hue='Survived', shade=True)
plt.show()

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}
train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)
test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)

plot_distribution('Embarked')

train_data['Fare'].fillna(train_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)
test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform('median'), inplace=True)

sns.kdeplot(data=train_data, x='Fare', hue='Survived', shade=True)
plt.show()

for dataset in [train_data, test_data]:
    dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')

cabin_mapping = {'A': 0, 'B': 0.4, 'C': 0.8, 'D': 1.2, 'E': 1.6, 'F': 2, 'G': 2.4, 'T': 2.8, 'Unknown': -1}
for dataset in [train_data, test_data]:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    dataset['Cabin'].fillna(train_data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

plot_distribution('Cabin')

for dataset in [train_data, test_data]:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

sns.kdeplot(data=train_data, x='FamilySize', hue='Survived', shade=True)
plt.show()

columns_to_drop = ['Ticket', 'SibSp', 'Parch', 'PassengerId']
train_data.drop(columns=columns_to_drop, inplace=True)
test_data.drop(columns=['Ticket', 'SibSp', 'Parch'], inplace=True)

X = train_data.drop(columns='Survived')
y = train_data['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.25, random_state=0)

models = {
    'RandomForest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=58),
    'DecisionTree': DecisionTreeClassifier(criterion='entropy', random_state=0),
    'SVM': SVC(kernel='rbf'),
    'NaiveBayes': GaussianNB()
}

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

for model_name, model in models.items():
    model.fit(X_train_imputed, y_train)
    
    y_pred = model.predict(X_val_imputed)
    
    accuracy = model.score(X_val_imputed, y_val) * 100
    
    print(f"{model_name} Accuracy: {accuracy:.2f}%")

test_data.drop(columns=['PassengerId'], inplace=True)
test_data_scaled = scaler.transform(test_data)
test_data_scaled = pd.DataFrame(test_data_scaled, columns=test_data.columns)

for model_name, model in models.items():
    test_data[f'Survived_{model_name}'] = model.predict(test_data_scaled)

test_data.to_csv('FinalResults.csv', index=False)















