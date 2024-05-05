import warnings
import streamlit as st
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
plt.rcParams['figure.figsize'] = [10, 6]
warnings.filterwarnings('ignore')

df = pd.read_csv("D:\\DLL\\archive\\song_data.csv")
df.head()

df.info()

df.describe()

df.drop(['song_name'], axis=1, inplace=True)
# df

target = 'song_popularity'
features = [i for i in df.columns if i not in [target]]
original_df = df.copy(deep=True)
df.info()

df.nunique().sort_values()

nu = df[features].nunique().sort_values()
nf = []
cf = []
nnf = 0
ncf = 0
for i in range(df[features].shape[1]):
    if nu.values[i] <= 16:
        cf.append(nu.index[i])
    else:
        nf.append(nu.index[i])
print("numerical features: ", nf)
print("\ncategorical features: ", cf)
print("\n\ntotal numerical features: ", len(nf),
      "\ntotal categorical features", len(cf))

plt.figure(figsize=[8, 4])
sns.distplot(df[target], color='b', hist_kws=dict(
    edgecolor="red", linewidth=2), bins=30)
plt.title('Target Variable Distribution')
plt.show()

sns.barplot(x='audio_mode', y='song_popularity', data=df)
plt.title('Popularity Based on Mode')
# 1-Minor 0-Major

sns.barplot(x='key', y='song_popularity', data=df)
plt.title('Popularity Based on key')
# 0-C# 1-D# 2-C 3-D  4-G# 5-F# 6-B 7-A 8-G 9-E 10-A# 11-F

sns.jointplot(x='loudness', y='song_popularity', data=df)

popular_above_50 = df[df.song_popularity > 50]
popular_below_50 = df[df.song_popularity < 50]
sns.distplot(popular_above_50['loudness'])
plt.title('Loudness for Songs with more than 50 Popularity')

sns.distplot(popular_below_50['loudness'])
plt.title('Loudness for Songs with less than 50 Popularity')

counter = 0
rs, cs = original_df.shape
df.drop_duplicates(inplace=True)
if df.shape == (rs, cs):
    print('No duplicates')
else:
    print("Number of duplicates", rs-df.shape[0])
nvc = pd.DataFrame(df.isnull().sum().sort_values(),
                   columns=['total null values'])
nvc['Percentage'] = round(nvc['total null values']/df.shape[0], 3)*100
print(nvc)

df3 = df.copy()
ecc = nvc[nvc['Percentage'] != 0].index.values
fcc = [i for i in cf if i not in ecc]
oh = True
dm = True
print("coverted features: ", end="")
for i in fcc:
    # print(i)
    if df3[i].nunique() == 2:
        print(i)
        oh = False
        df3[i] = pd.get_dummies(df3[i], drop_first=True, prefix=str(i))
    if (df3[i].nunique() > 2 and df3[i].nunique() < 17):
        print(i)
        dm = False
        df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(
            pd.get_dummies(df3[i], drop_first=True, prefix=str(i)))], axis=1)
df3.shape

print('\033[1mOutliers'.center(150))
n = 5
plt.figure(figsize=[15, 4*math.ceil(len(nf)/n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf)/3), n, i+1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()

df1 = df3.copy()
features1 = nf
for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3-Q1
    # interquartile range=diff b/w 3rd and 1st quartile
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop=True)
    # any values that are less than Q1-1.5*IQR or greater than Q3+1.5*IQR are considered outliers and are removed
print("before removal: ", df3.shape[0])
print("after removal: ", df1.shape[0])


m = []
for i in df.columns.values:
    m.append(i.replace(' ', '_'))

df.columns = m
X = df.drop(columns=["song_popularity"], axis=1)
Y = df[target]  # popularity
Train_X, Test_X, Train_Y, Test_Y = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True, inplace=True)
# print('\nTraining set: ',Train_X.shape,Train_Y.shape,'\nTesting set: ', Test_X.shape,'', Test_Y.shape)
# print(X)
print(Y)

std = StandardScaler()
print('Standardardization on Training set'.center(120))
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
display(Train_X_std.describe())
print('\n', 'Standardardization on Testing set'.center(120))
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
display(Test_X_std.describe())


Model_Evaluation_Comparison_Matrix = pd.DataFrame(np.zeros([5, 8]), columns=['Train-R2', 'Test-R2', 'Train-RSS', 'Test-RSS',
                                                                             'Train-MSE', 'Test-MSE', 'Train-RMSE', 'Test-RMSE'])
rc = np.random.choice(
    Train_X_std.loc[:, Train_X_std.nunique() >= 50].columns.values, 3, replace=False)


def Evaluate(n, pred1, pred2):
    plt.figure(figsize=[15, 6])
    for e, i in enumerate(rc):
        plt.subplot(2, 3, e+1)
        plt.scatter(y=Train_Y, x=Train_X_std[i], label='Actual')
        plt.scatter(y=pred1, x=Train_X_std[i], label='Prediction')
        plt.legend()
    plt.show()

    print('\n\n{}Training Set Metrics{}'.format('-'*20, '-'*20))
    print('\nR2-Score on Training set: ', round(r2_score(Train_Y, pred1), 20))
    print('Residual Sum of Squares: ', round(
        np.sum(np.square(Train_Y-pred1)), 20))
    print('Mean Squared Error: ', round(mean_squared_error(Train_Y, pred1), 20))
    print('Root Mean Squared Error: ', round(
        np.sqrt(mean_squared_error(Train_Y, pred1)), 20))

    print('\n{}Testing Set Metrics{}'.format('-'*20, '-'*20))
    print('\nR2-Score on Testing set : ', round(r2_score(Test_Y, pred2), 20))
    print('Residual Sum of Squares: ', round(
        np.sum(np.square(Test_Y-pred2)), 20))
    print('Mean Squared Error: ', round(mean_squared_error(Test_Y, pred2), 20))
    print('Root Mean Squared Error: ', round(
        np.sqrt(mean_squared_error(Test_Y, pred2)), 20))
    print('\n{}Residual Plots{}'.format('-'*20, '-'*20))

    Model_Evaluation_Comparison_Matrix.loc[n,
                                           'Train-R2'] = round(r2_score(Train_Y, pred1), 20)
    Model_Evaluation_Comparison_Matrix.loc[n,
                                           'Test-R2'] = round(r2_score(Test_Y, pred2), 20)
    Model_Evaluation_Comparison_Matrix.loc[n, 'Train-RSS'] = round(
        np.sum(np.square(Train_Y-pred1)), 20)
    Model_Evaluation_Comparison_Matrix.loc[n,
                                           'Test-RSS'] = round(np.sum(np.square(Test_Y-pred2)), 20)
    Model_Evaluation_Comparison_Matrix.loc[n, 'Train-MSE'] = round(
        mean_squared_error(Train_Y, pred1), 20)
    Model_Evaluation_Comparison_Matrix.loc[n, 'Test-MSE'] = round(
        mean_squared_error(Test_Y, pred2), 20)
    Model_Evaluation_Comparison_Matrix.loc[n, 'Train-RMSE'] = round(
        np.sqrt(mean_squared_error(Train_Y, pred1)), 20)
    Model_Evaluation_Comparison_Matrix.loc[n, 'Test-RMSE'] = round(
        np.sqrt(mean_squared_error(Test_Y, pred2)), 20)

    # Plotting y_test and y_pred to understand the spread.
    plt.figure(figsize=[15, 4])

    plt.subplot(1, 2, 1)
    sns.distplot((Train_Y - pred1))
    plt.title('Error Terms')
    plt.xlabel('Errors')

    plt.subplot(1, 2, 2)
    plt.scatter(Train_Y, pred1)
    plt.plot([Train_Y.min(), Train_Y.max()], [
             Train_Y.min(), Train_Y.max()], 'r--')
    plt.title('Test vs Prediction')
    plt.xlabel('y_test')
    plt.ylabel('y_pred')
    plt.show()


model1 = RandomForestClassifier(n_estimators=10, criterion='entropy')
model1.fit(Train_X_std, Train_Y)
train_pred = model1.predict(Train_X_std)
test_pred = model1.predict(Test_X_std)
score1 = accuracy_score(Test_Y, test_pred)
score2 = accuracy_score(Train_Y, train_pred)
print("Tetsing Accuracy for RF: ", score1)
print("Testing Accuracy for RF: ", score2)

print(test_pred)

user_input = [[262333, 0.00552, 0.496, 0.682, 2.94E-05, 8, 0.0589, -4.095, 1, 0.0294, 167.06, 4, 0.474
               ]]
user_input_df = pd.DataFrame(data=user_input)
user_input_std = std.transform(user_input)
user_input_std_df = pd.DataFrame(
    data=user_input_std, columns=user_input_df.columns)
pred_user_std = model1.predict(user_input_std_df)
print(pred_user_std)
st.title("Song Popularity Predictor")

st.write("Enter song features to predict its popularity:")
user_input = st.text_area("Enter song features (comma-separated)",
                          "262333,0.00552,0.496,0.682,2.94E-05,8,0.0589,-4.095,1,0.0294,167.06,4,0.474")

if st.button("Predict"):
    user_input = [[float(x) for x in user_input.split(",")]]
    user_input_std = std.transform(user_input)
    user_input_std_df = pd.DataFrame(data=user_input_std, columns=X.columns)
    pred_user_std = model1.predict(user_input_std_df)
    st.write("Predicted popularity class:", pred_user_std[0])
