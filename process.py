import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

print("Step 1: Loading and Exploring Data")
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
df = pd.read_csv(url)

print("--- Initial Dataset Info ---")
df.info()

print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())
print("Note: This dataset has no missing values, so no filling is needed.")
print("-" * 40)


print("\nStep 2: Encoding Categorical Features")

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("'Sex' column encoded to numerical (0: male, 1: female).")

print("\n--- Data Types After Encoding ---")
df.info()
print("-" * 40)



print("\nStep 3: Handling Outliers in 'Age' and 'Fare'")


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['Fare'])
plt.title('Boxplot of Fare')


plt.savefig('outlier_boxplots.png')
print("Boxplots saved as 'outlier_boxplots.png'")
plt.show()


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_rows = len(df)
    df_out = df.loc[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"Removed {initial_rows - len(df_out)} outliers from '{column}'.")
    return df_out


df.rename(columns={
    'Siblings/Spouses Aboard': 'SibSp',
    'Parents/Children Aboard': 'Parch'
}, inplace=True)



df = remove_outliers(df, 'Age')
df = remove_outliers(df, 'Fare')

print("\n--- Shape of Data After Removing Outliers ---")
print(df.shape)
print("-" * 40)



print("\nStep 4: Standardizing Numerical Features")


df.drop(['Name'], axis=1, inplace=True)
print("Dropped 'Name' column.")


X = df.drop('Survived', axis=1)
y = df['Survived']

numerical_cols = X.select_dtypes(include=np.number).columns

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\n--- First 5 Rows of Standardized Features ---")
print(X.head())

print("\n--- Preprocessing Complete! ---") 
