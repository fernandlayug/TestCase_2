import graphviz
from IPython.display import Image

# Create a Digraph object
graph = graphviz.Digraph()

# Define nodes and edges based on the model evaluation process
graph.node('A', 'Data Loading')

# Data Preprocessing subprocesses
with graph.subgraph(name='cluster_data_preprocessing') as sub:
    sub.node('B', 'Data Preprocessing')
    sub.node('B1', 'Separate features\nand target variable')
    sub.node('B2', 'Identify ordinal features')
    sub.node('B3', 'Scale non-ordinal features\nusing StandardScaler')
    sub.node('B4', 'Encode ordinal features\nusing OrdinalEncoder')
    sub.node('B5', 'Combine scaled and encoded features')

graph.node('C', 'Model Training\nand Hyperparameter Tuning')

# Model Training subprocesses
with graph.subgraph(name='cluster_model_training') as sub:
    sub.node('C1', 'Define XGBoost Classifier')
    sub.node('C2', 'Define hyperparameters\ngrid for GridSearchCV')
    sub.node('C3', 'Perform GridSearchCV\nfor hyperparameter tuning')
    sub.node('C4', 'Get best hyperparameters')
    sub.node('C5', 'Train XGBoost model\nwith best hyperparameters')

graph.node('D', 'Model Evaluation')

# Model Evaluation subprocesses
with graph.subgraph(name='cluster_model_evaluation') as sub:
    sub.node('D1', 'K-fold Cross-validation')
    sub.node('D2', 'Evaluate on testing set:')
    sub.node('D2A', 'Accuracy')
    sub.node('D2B', 'Confusion Matrix')
    sub.node('D2C', 'Classification Report')
    sub.node('D2D', 'ROC AUC Curve and Score')

graph.node('E', 'Feature Importance Analysis')

# Feature Importance Analysis subprocesses
with graph.subgraph(name='cluster_feature_importance') as sub:
    sub.node('E1', 'Calculate feature importances')
    sub.node('E2', 'Visualize feature importances\nusing a bar plot')

graph.node('F', 'SHAP Values Analysis')

# SHAP Values Analysis subprocesses
with graph.subgraph(name='cluster_shap_values') as sub:
    sub.node('F1', 'Compute SHAP values')
    sub.node('F2', 'Visualize SHAP summary plot')
    sub.node('F3', 'Visualize SHAP dependence plots\nfor each feature')

graph.node('G', 'Visualization of\nEvaluation Metrics')

# Visualization of Evaluation Metrics subprocesses
with graph.subgraph(name='cluster_evaluation_metrics') as sub:
    sub.node('G1', 'Visualize cross-validation\nscores using a bar plot')
    sub.node('G2', 'Visualize confusion matrix\nfor the testing set\nusing a heatmap')
    sub.node('G3', 'Plot ROC Curve')

graph.node('H', 'Hyperparameter\nEffect Analysis')

# Hyperparameter Effect Analysis subprocesses
with graph.subgraph(name='cluster_hyperparameter_effect') as sub:
    sub.node('H1', 'Visualize the effect of\nreg_alpha on cross-validated\nperformance')
    sub.node('H2', 'Visualize the effect of\nreg_lambda on cross-validated\nperformance')

# Define edges for the main steps
graph.edges([
    ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'),
    ('E', 'F'), ('F', 'G'), ('G', 'H')
])

# Define edges for subprocesses
graph.edge('B', 'B1')
graph.edge('B', 'B2')
graph.edge('B', 'B3')
graph.edge('B', 'B4')
graph.edge('B', 'B5')

graph.edge('C', 'C1')
graph.edge('C', 'C2')
graph.edge('C', 'C3')
graph.edge('C', 'C4')
graph.edge('C', 'C5')

graph.edge('D', 'D1')
graph.edge('D', 'D2')
graph.edge('D2', 'D2A')
graph.edge('D2', 'D2B')
graph.edge('D2', 'D2C')
graph.edge('D2', 'D2D')

graph.edge('E', 'E1')
graph.edge('E', 'E2')

graph.edge('F', 'F1')
graph.edge('F', 'F2')
graph.edge('F', 'F3')

graph.edge('G', 'G1')
graph.edge('G', 'G2')
graph.edge('G', 'G3')

graph.edge('H', 'H1')
graph.edge('H', 'H2')

# Render the flowchart
flowchart_path = 'model_evaluation_flowchart_with_subprocesses.png'
graph.render(flowchart_path, format='png', cleanup=True)

# Display the flowchart
Image(filename=flowchart_path)
