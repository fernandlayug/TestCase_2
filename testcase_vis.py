from graphviz import Digraph

# Create a new Digraph object
dot = Digraph(comment='Model Evaluation Flowchart')

# Start symbol
dot.node('Start', 'Start', shape='circle')

# Step 1: Data Loading and Preparation
dot.node('A', 'Data Loading and Preparation', shape='parallelogram')
dot.node('A1', "Read Data from 'selected_data_top_contributing_features_1.xlsx'", shape='box')
dot.node('A2', 'Define Ordinal and Non-Ordinal Features', shape='box')
dot.node('A3', 'Split into Train/Test Sets\n(train_test_split)', shape='box')

dot.edges([('Start', 'A')])
dot.edges([('A', 'A1'), ('A', 'A2'), ('A', 'A3')])

# Decision symbol for Data Preprocessing
dot.node('B', 'Data Preprocessing?', shape='diamond')
dot.edges([('A1', 'B'), ('A2', 'B'), ('A3', 'B')])

# Step 2: Data Preprocessing
dot.node('B1', 'Scale Non-Ordinal Features\n(StandardScaler)', shape='box')
dot.node('B2', 'Transform X_test\n(StandardScaler)', shape='box')
dot.node('B3', 'Encode Ordinal Features\n(OrdinalEncoder)', shape='box')
dot.node('B4', 'Transform X_test\n(OrdinalEncoder)', shape='box')
dot.node('B5', 'Combine Features\n(Concatenate)', shape='box')
dot.node('B6', 'Save Scalers and Encoder', shape='box')

dot.edges([('B', 'B1')])
dot.edges([('B1', 'B2'), ('B2', 'B3')])
dot.edges([('B3', 'B4'), ('B4', 'B5')])
dot.edges([('B5', 'B6')])

# Decision symbol for Model Training
dot.node('C', 'Model Training?', shape='diamond')
dot.edges([('B6', 'C')])

# Step 3: Model Training and Hyperparameter Tuning
dot.node('C1', 'Define XGBoost\n(XGBClassifier)', shape='box')
dot.node('C2', 'Define Hyperparameters Grid\n(param_grid)', shape='box')
dot.node('C3', 'Perform GridSearchCV\n(GridSearchCV)', shape='box')
dot.node('C4', 'Get Best Parameters\n(best_params)', shape='box')
dot.node('C5', 'Train Model with Best Params\n(best_xgb)', shape='box')
dot.node('C6', 'Save Trained Model\n(joblib)', shape='box')

param_grid_values = (
    "param_grid = {\n"
    "  'n_estimators': [50, 100, 150],\n"
    "  'learning_rate': [0.05, 0.1, 0.15],\n"
    "  'max_depth': [3, 4, 5, 6, 7],\n"
    "  'min_child_weight': [1, 2, 4],\n"
    "  'subsample': [0.7, 0.8, 0.9, 1.0],\n"
    "  'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],\n"
    "  'reg_alpha': [0, 0.1, 0.5, 1],\n"
    "  'reg_lambda': [0.5, 1, 1.5, 2]\n"
    "}"
)

dot.node('C2A', param_grid_values, shape='box')

dot.edges([('C', 'C1')])
dot.edges([('C1', 'C2'), ('C2', 'C3')])
dot.edges([('C3', 'C4'), ('C4', 'C5')])
dot.edges([('C5', 'C6'), ('C2A', 'C3')])

# Decision symbol for Model Evaluation
dot.node('D', 'Model Evaluation?', shape='diamond')
dot.edges([('C6', 'D')])

# Step 4: Model Evaluation
dot.node('D1', 'Predict on Test Set\n(best_xgb)', shape='box')
dot.node('D2', 'Accuracy Score\n(accuracy_score)', shape='box')
dot.node('D3', 'Confusion Matrix\n(confusion_matrix)', shape='box')
dot.node('D4', 'Classification Report\n(classification_report)', shape='box')
dot.node('D5', 'ROC AUC Score\n(roc_auc_score)', shape='box')
dot.node('D6', 'ROC Curve\n(roc_curve)', shape='box')

dot.edges([('D', 'D1')])
dot.edges([('D1', 'D2'), ('D2', 'D3')])
dot.edges([('D3', 'D4'), ('D4', 'D5')])
dot.edges([('D5', 'D6')])

# Step 5: Model Interpretation
dot.node('E', 'Model Interpretation', shape='parallelogram')
dot.node('E1', 'Feature Importance\n(best_xgb)', shape='box')
dot.node('E2', 'Visualize Feature Importances\n(Seaborn)', shape='box')
dot.node('E3', 'SHAP Values Analysis\n(shap.Explainer)', shape='box')
dot.node('E4', 'SHAP Summary Plot\n(shap.summary_plot)', shape='box')
dot.node('E5', 'SHAP Dependence Plot\n(shap.dependence_plot)', shape='box')

dot.edges([('D', 'E'), ('E1', 'E'), ('E2', 'E')])
dot.edges([('E3', 'E'), ('E4', 'E'), ('E5', 'E')])

# Step 6: Model Validation
dot.node('F', 'Model Validation', shape='parallelogram')
dot.node('F1', 'Cross-Validation\n(KFold: n_splits=5, random_state=42)', shape='box')
dot.node('F2', 'Visualize CV Scores\n(Seaborn)', shape='box')
dot.node('F3', 'Visualize Effect of reg_alpha\nand reg_lambda\n(Seaborn)', shape='box')

dot.edges([('E', 'F'), ('F1', 'F'), ('F2', 'F')])
dot.edges([('F3', 'F')])

# End symbol
dot.node('End', 'End', shape='circle')
dot.edges([('F2', 'End'), ('F3', 'End'), ('D6', 'End')])

# Render the graph
dot.render('model_evaluation_flowchart', format='png', view=True)
