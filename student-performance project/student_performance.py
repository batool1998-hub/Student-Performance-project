
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, r2_score

# ===== Load and Preprocess Data =====
df = pd.read_csv('Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

X = df.drop('Performance Index', axis=1)
y = df['Performance Index']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Save Preprocessed Data =====
os.makedirs('Data/preprocessed_data', exist_ok=True)
X.to_csv('Data/preprocessed_data/X.csv', index=False)
X_test.to_csv('Data/preprocessed_data/X_test.csv', index=False)
y.to_csv('Data/preprocessed_data/Y.csv', index=False)
y_test.to_csv('Data/preprocessed_data/Y_test.csv', index=False)

# ===== Define Models =====
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVM': SVR(),
    'KNN': KNeighborsRegressor(),
    'ANN': MLPRegressor(random_state=42, max_iter=500),
    'Naive Bayes': GaussianNB()
}

results = []
predictions = {}

# ===== Train, Predict, Evaluate, Save Results =====
os.makedirs('Results', exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save first 10 predictions
    df_result = pd.DataFrame({
        'Actual': y_test[:10].values,
        'Predicted': y_pred[:10]
    })
    predictions[name] = df_result
    filename = name.lower().replace(" ", "_").replace("-", "") + '_results.csv'
    df_result.to_csv(f'Results/{filename}', index=False)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MSE': round(mse, 2), 'R2 Score': round(r2, 3)})

# ===== Save Evaluation Summary =====
results_df = pd.DataFrame(results)
results_df.to_csv('Results/model_performance_summary.csv', index=False)

# ===== Visualize MSE =====
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='MSE', data=results_df, palette='Set2')
plt.title('Mean Squared Error (MSE) Comparison')
plt.ylabel('MSE')
plt.xticks(rotation=45)
for i, val in enumerate(results_df['MSE']):
    plt.text(i, val + 0.5, round(val, 2), ha='center')
plt.tight_layout()
plt.savefig('Results/MSE_comparison.png')
plt.close()

# ===== Visualize R² Score =====
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='R2 Score', data=results_df, palette='Set3')
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
for i, val in enumerate(results_df['R2 Score']):
    plt.text(i, val + 0.01, round(val, 3), ha='center')
plt.tight_layout()
plt.savefig('Results/R2_score_comparison.png')
plt.close()
