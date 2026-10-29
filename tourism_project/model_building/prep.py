# for data manipulation
import pandas as pd
import sklearn
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = "tourism_project/data/tourism.csv"

# A small number of rows contain an unescaped embedded quote in app_name,
# which breaks CSV tokenization; skip just those rows rather than failing.
df = pd.read_csv(DATASET_PATH, on_bad_lines="skip")
print("Dataset loaded successfully.")

# Drop unique identifier column (not useful for modeling)
if 'CustomerID' in df.columns:
    df.drop(columns=['CustomerID'], inplace=True)
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns identified for encoding:", categorical_cols)

#Update Gender column where Fe Male to Female
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

#Update MaritialStatue column where Unmarried to Single
df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')

# Encode categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    # Fill missing values in categorical columns with a placeholder before encoding
    df[col] = df[col].fillna('Unknown').astype(str)
    df[col] = label_encoder.fit_transform(df[col])

# Define target variable
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written with categorical encoding applied.")
