# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the Titanic dataset
# Using seaborn's built-in Titanic dataset
import seaborn as sns
data = sns.load_dataset("titanic")

# Step 2: Data Exploration
print("Dataset Head:\n", data.head())
print("\nDataset Info:\n")
data.info()

# Step 3: Define target and features
target = "survived"
features = ["pclass", "sex", "age", "fare", "embarked"]

X = data[features]
y = data[target]

# Step 4: Data Preprocessing
# Define preprocessing steps for numerical and categorical features
numerical_features = ["age", "fare"]
categorical_features = ["pclass", "sex", "embarked"]

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),  # Handle missing values
    ("scaler", StandardScaler())  # Standardize numerical values
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Handle missing values
    ("encoder", OneHotEncoder(handle_unknown="ignore"))  # Convert to one-hot encoding
])

# Combine numerical and categorical transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Step 5: Build the Pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),  # Apply transformations
    ("classifier", LogisticRegression())  # Classifier
])

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Pipeline
pipeline.fit(X_train, y_train)

# Step 8: Evaluate the Model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Step 9: Save Processed Data (Optional)
X_processed = pipeline.named_steps["preprocessor"].transform(X)

# Convert the processed data to a DataFrame
from scipy.sparse import issparse

# If the result is a sparse matrix, convert it to a dense array
if issparse(X_processed):
    X_processed = X_processed.toarray()

processed_data = pd.DataFrame(X_processed)
processed_data.to_csv("processed_titanic_data.csv", index=False)

print("\nPipeline execution completed. Processed data saved.")
