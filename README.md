# DATA-PIPELINE-DEVELOPMENT

**Company:** CODTECH IT SOLUTIONS PVT. LTD.

**Name:** KEYUR AMIT CHAUHAN

**Intern ID:** CT08EOQ

**Domain:** DATA SCIENCE

**Duration:** December 17th, 2024 - January 17th, 2025 (4 weeks)

**Mentor:** Neela Santhosh Kumar,HR & Academic Head

In this task, a comprehensive data pipeline was created to preprocess, transform, and load data using tools like Pandas and Scikit-learn. The dataset utilized was the Titanic dataset, which includes information about passengers such as age, gender, class, and survival status.

Data Loading and Inspection:
The pipeline began with loading the dataset into a Pandas DataFrame, followed by inspecting the data structure using methods like .head(), .info(), and .describe(). This step provided an overview of data types, null values, and statistical properties of numerical columns.

Data Preprocessing:

Handling Missing Values: Columns like age and embarked had missing values, which were imputed using median and mode values, respectively.
Encoding Categorical Features: Non-numeric columns such as sex and embarked were converted into numerical format using one-hot encoding.
Feature Scaling: Numerical columns were scaled using Scikit-learn’s StandardScaler to normalize the data for model training.
Feature Selection and Transformation:
Irrelevant columns, such as passenger names or ticket numbers, were dropped to focus on features contributing to prediction. The remaining data was split into independent variables (X) and the target variable (y), which represented survival status.

Data Splitting:
The dataset was divided into training and testing sets using Scikit-learn’s train_test_split function, ensuring an 80-20 ratio. This split allowed for model evaluation on unseen data.

Model Training and Evaluation:
A logistic regression model was trained using the processed training data. After training, the model was evaluated on the test data, achieving an accuracy score of 79%, indicating satisfactory performance.

Pipeline Automation:
The entire ETL (Extract, Transform, Load) process was automated using a Python script, encapsulating all steps from data loading to model evaluation. This automation ensures consistency and repeatability for similar datasets in the future.

This project showcased the implementation of an efficient data pipeline for preprocessing and transformation, streamlining the data preparation phase for machine learning tasks.
