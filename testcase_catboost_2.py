import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Step 1: Load the dataset
def load_dataset(file_path):
    data = pd.read_excel(file_path)
    return data

# Step 2: Prepare the data
def prepare_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    # Encode categorical features if needed
    categorical_features_indices = [i for i, col in enumerate(X.columns) if X[col].dtype == 'object']
    
    return X, y, categorical_features_indices

# Step 3: Split the data
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Step 4: Hyperparameter tuning using GridSearchCV
def tune_hyperparameters(X_train, y_train, categorical_features_indices):
    model = CatBoostClassifier(cat_features=categorical_features_indices, verbose=0)
    
    # Define the parameter grid
    param_grid = {
        'iterations': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8],
        'l2_leaf_reg': [1, 3, 5, 7],
        'border_count': [32, 64, 128]
    }
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Best Hyperparameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Step 5: Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC AUC
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    return accuracy, report, cm, roc_auc, y_test, y_prob

# Step 6: Plot the confusion matrix
def plot_confusion_matrix(cm, target_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Step 7: Plot ROC AUC
def plot_roc_auc(y_test, y_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

# Main function to execute the workflow
def main():
    # Replace 'your_dataset.xlsx' with the path to your Excel file
    file_path = 'datasets/selected_data_top_contributing_features_1.xlsx'
    target_column = 'Completed'  # Replace 'target' with the actual target column name

    data = load_dataset(file_path)
    X, y, categorical_features_indices = prepare_data(data, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)

    best_model = tune_hyperparameters(X_train, y_train, categorical_features_indices)
    accuracy, report, cm, roc_auc, y_test, y_prob = evaluate_model(best_model, X_test, y_test)

    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)
    print(f'ROC AUC: {roc_auc}')

    # Plotting
    target_names = y.unique()  # Assuming binary classification, use `y.unique()` for multi-class
    plot_confusion_matrix(cm, target_names)
    plot_roc_auc(y_test, y_prob)

if __name__ == "__main__":
    main()
