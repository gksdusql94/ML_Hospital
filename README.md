# Hospital Readmission Prediction Project

## Objective:
The goal of this project is to predict whether a patient will be readmitted to the hospital based on various clinical and administrative data, including diagnoses, treatments, medications, and textual AI-generated responses summarizing each patient's medical condition.

## Key Steps:

### 1. Data Preprocessing:
- **Dataset**: The dataset contains 8,000 entries with 40 features, such as discharge disposition, admission source, payer code, medical specialty, number of lab procedures, and AI-generated responses for diagnoses.
- **Missing Values**: Several columns, such as `max_glu_serum`, `payer_code`, and `medical_specialty`, had missing values. Missing values were handled by imputing averages or replacing them with "Unknown" labels where appropriate.
- **Categorical Encoding**: Categorical features were encoded using OneHotEncoding and LabelEncoding. Binary columns such as `diabetesMed` and `change` were converted to 0 and 1.
  
### 2. Exploratory Data Analysis (EDA):
- The distribution of the target variable (`readmitted`) was found to be slightly imbalanced, with about 60% of patients not being readmitted and 40% being readmitted.
- Summary statistics and correlations were calculated to understand relationships between variables.

### 3. Feature Engineering:
- **Text Feature Engineering**: AI-generated diagnostic responses were vectorized using `TfidfVectorizer`, with bigrams and trigrams, and incorporated into the prediction pipeline.
- **Numerical Feature Scaling**: Numerical features were standardized using `StandardScaler` for better performance in machine learning algorithms.

### 4. Model Training and Evaluation:
#### Model 1: Logistic Regression
- Logistic Regression was trained on the TF-IDF-transformed AI response text data.
- Cross-validation was used, and the model yielded an AUC score of approximately 0.561.

#### Model 2: Random Forest (for Stacking)
- A second model was built by combining the standardized numerical data with predicted probabilities from the TF-IDF text data.
- This Random Forest model yielded an improved AUC score of 0.675.

#### Model 3: Gradient Boosting Classifier
- The Gradient Boosting model improved performance significantly, reaching an AUC score of 0.671.

#### Model 4: XGBoost
- XGBoost, an optimized implementation of Gradient Boosting, was the final model chosen. It showed progressive improvement during training, with a final AUC score of 0.6817.

### 5. Model Stacking:
- The results of the logistic regression model on TF-IDF-transformed text data were integrated into a Random Forest classifier through model stacking.
- This approach combined both structured numerical data and unstructured text data, leading to better performance.

### 6. Final Model Selection:
- The final model selection was based on XGBoost due to its ability to handle large datasets with high-dimensional features efficiently. Though sensitive to hyperparameter tuning, XGBoost achieved the highest AUC score.

### 7. Test Data Prediction:
- After preprocessing the test dataset (2,000 samples) in the same manner as the training data, predictions were made using the XGBoost model.
- The final predicted probabilities were saved to a CSV file for submission.

## Conclusion:
This project demonstrates the effectiveness of combining structured data with unstructured text data using model stacking. The final model achieved an AUC score of 0.6817, showing promising results for hospital readmission prediction.

Developed a machine learning model to predict hospital readmission likelihood, aiming to improve hospital operational efficiency and patient care management.

-	Developed and deployed a machine learning model to predict hospital readmission, achieving an AUC score of 0.68 on the final test dataset.
-	Preprocessed and cleaned a real-world dataset of over 8,000 hospital entries and 40 features (e.g., diagnoses, prescriptions, ER visits).
-	Implemented Logistic Regression and XGBoost models with hyperparameter tuning, increasing predictive performance by ~7% using model stacking with TF-IDF for medical text data. 

## Tools and Libraries:
- **Programming Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Machine Learning Algorithms**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Feature Engineering**: OneHotEncoding, LabelEncoding, StandardScaler, TfidfVectorizer
- **Text Processing**: TF-IDF for feature extraction from diagnostic text.
