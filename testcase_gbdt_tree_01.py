import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
from xgboost import plot_tree
import graphviz

# Load the trained model
best_xgb = joblib.load('model/best_xgb_model_17.pkl')

# Create a plot of the first tree in the ensemble
plt.figure(figsize=(40, 20))  # Larger size for better resolution
plot_tree(best_xgb, num_trees=0, rankdir='LR')
plt.savefig('tree_plot.png', dpi=300, bbox_inches='tight')  # Save the figure as a PNG file with high resolution
plt.show()

# Export the tree as a dot file
xgb.to_graphviz(best_xgb, num_trees=0, rankdir='LR', size='10,10').render('tree_plot', format='pdf')
