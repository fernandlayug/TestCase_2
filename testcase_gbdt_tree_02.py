import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import graphviz

# Load the trained model
best_xgb = joblib.load('model/best_xgb_model_17.pkl')

# Define feature names (replace with your actual feature names)
feature_names = ['ParentCohabitation_Widowed', 'SchoolAttended_Private', 'ReasonStopSchool_CannotCope', 'FinancialSupport', 'AttendClassRegularly', 'ReasonStopSchool_SchoolsFar', 'ParentCohabitation_LivingTogether', 'TravelTime', 'InternetAccess', 'MotherJob_OtherJob', 'DistanceHomeSchool', 'FatherJob', 'PerformanceScale', 'DaysAvailable', 'Gender', 'Guardian_Mother', 'SchoolAttended_Public']

# Create a plot of the first tree in the ensemble using Matplotlib
plt.figure(figsize=(40, 20))  # Larger size for better resolution
xgb.plot_tree(best_xgb, num_trees=0, rankdir='LR', feature_names=feature_names)
plt.savefig('tree_plot.png', dpi=300, bbox_inches='tight')  # Save the figure as a PNG file with high resolution
plt.show()

# Export the tree as a dot file and render it using Graphviz
dot_data = xgb.to_graphviz(best_xgb, num_trees=0, rankdir='LR', size='10,10', feature_names=feature_names)
graph = graphviz.Source(dot_data)
graph.render('tree_plot', format='pdf')
