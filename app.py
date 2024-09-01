import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

@st.cache_resource
def load_model():
    model = SimpleCpGPredictor()
    model.load_state_dict(torch.load('cpg_detector_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_sequence(sequence):
    alphabet = 'NACGT'
    dna2int = {a: i for i, a in enumerate(alphabet)}
    int_seq = [dna2int.get(base, 0) for base in sequence.upper()]
    one_hot = F.one_hot(torch.tensor(int_seq), num_classes=5).float()
    return one_hot

def normalize_data(data):
    return (data - data.mean()) / data.std()

def predict_cpg_count(model, sequence):
    with torch.no_grad():
        input_tensor = preprocess_sequence(sequence)
        input_tensor = normalize_data(input_tensor)
        
        # Pad or truncate the input to match the expected input size
        expected_length = 128
        if input_tensor.size(0) < expected_length:
            padding = torch.zeros(expected_length - input_tensor.size(0), 5)
            input_tensor = torch.cat([input_tensor, padding], dim=0)
        elif input_tensor.size(0) > expected_length:
            input_tensor = input_tensor[:expected_length]
        
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
    return output.item()

def main():
    st.title("CpG Count Predictor")

    st.write("This app predicts the number of CpG sites in a given DNA sequence.")

    model = load_model()

    sequence = st.text_input("Enter a DNA sequence:", "")

    if st.button("Predict"):
        if len(sequence) < 1:
            st.error("Please enter a valid DNA sequence.")
        else:
            prediction = predict_cpg_count(model, sequence)
            st.success(f"Predicted CpG count: {prediction:.2f}")

    st.write("Note: This model can handle DNA sequences of any length, but was trained on sequences of 128 bases. For best results, use sequences close to this length.")

if __name__ == "__main__":
    main()
