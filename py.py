import pickle

model_path = "model/rf_model.pkl"  # Adjust the path if necessary

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
