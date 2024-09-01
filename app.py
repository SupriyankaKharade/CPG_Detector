import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the model architecture to match the trained model
class SimpleCpGPredictor(nn.Module):
    def __init__(self, input_size=5*128, hidden_size=64, output_size=1):
        super(SimpleCpGPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    model = SimpleCpGPredictor()
    model.load_state_dict(torch.load('cpg_detector_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess input sequence
def preprocess_sequence(sequence):
    alphabet = 'NACGT'
    dna2int = {a: i for i, a in enumerate(alphabet)}
    int_seq = [dna2int.get(base, 0) for base in sequence.upper()]
    one_hot = F.one_hot(torch.tensor(int_seq), num_classes=5).float()
    return one_hot.unsqueeze(0)

# Make prediction
def predict_cpg_count(model, sequence):
    with torch.no_grad():
        input_tensor = preprocess_sequence(sequence)
        output = model(input_tensor)
    return output.item()

# Streamlit app
def main():
    st.title("CpG Count Predictor")

    st.write("This app predicts the number of CpG sites in a given DNA sequence.")

    # Load the model
    model = load_model()

    # Input text box for DNA sequence
    sequence = st.text_input("Enter a DNA sequence (128 bases):", "")

    if st.button("Predict"):
        if len(sequence) != 128:
            st.error("Please enter a sequence of exactly 128 bases.")
        else:
            prediction = predict_cpg_count(model, sequence)
            st.success(f"Predicted CpG count: {prediction:.2f}")

    st.write("Note: This model was trained on sequences of length 128. For best results, please enter sequences of this length.")

if __name__ == "__main__":
    main()
