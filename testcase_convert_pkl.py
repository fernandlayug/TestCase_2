import pickle
from Orange.classification import SklModel
from Orange.data.io import PickleWriter

# Load the model from .pkl
with open('model/best_xgb_model_17.pkl', 'rb') as f:
    skl_model = pickle.load(f)

# Wrap it in Orange's SklModel
orange_model = SklModel(skl_model)

# Save it to .pkcls using PickleFormat
output_path = 'your_model.pkcls'
pickle_format = PickleFormat()
pickle_format.write(orange_model, output_path)

print(f"Model saved to {output_path}")

print(f"Model saved to {output_path}")
