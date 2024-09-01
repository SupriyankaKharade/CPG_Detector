import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the CpGPredictor model class
class CpGPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        logits = self.classifier(last_output)
        return logits

# Helper functions
alphabet = 'NACGT'
dna2int = {a: i for i, a in enumerate(alphabet)}

def one_hot_encode(sequence, num_classes=5):
    sequence_tensor = torch.tensor([dna2int[base] for base in sequence], dtype=torch.long)
    one_hot = F.one_hot(sequence_tensor, num_classes=num_classes)
    return one_hot.float().unsqueeze(0)

# Load the trained model
@st.cache_resource
def load_model():
    model = CpGPredictor()
    model.load_state_dict(torch.load('cpg_detector_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Streamlit app
st.title('CpG Detector')

# Input DNA sequence
dna_sequence = st.text_input('Enter a DNA sequence (N, A, C, G, T):', 'NCACANNTNCGGAGGCGNA')

# Validate input
if not all(base in 'NACGT' for base in dna_sequence):
    st.error('Invalid input. Please use only N, A, C, G, T.')
else:
    # Load model
    model = load_model()

    # Predict CpG count
    with torch.no_grad():
        input_tensor = one_hot_encode(dna_sequence)
        prediction = model(input_tensor)

    # Display results
    st.write(f'Input DNA sequence: {dna_sequence}')
    st.write(f'Predicted CpG count: {prediction.item():.2f}')

    # Count actual CpGs for comparison
    actual_cpg_count = sum(dna_sequence[i:i+2] == "CG" for i in range(len(dna_sequence) - 1))
    st.write(f'Actual CpG count: {actual_cpg_count}')

    # Calculate and display error
    error = abs(prediction.item() - actual_cpg_count)
    st.write(f'Absolute error: {error:.2f}')

# Instructions
st.markdown("""
### Instructions:
1. Enter a DNA sequence using only the characters N, A, C, G, and T.
2. The app will display the predicted CpG count from the model.
3. For comparison, the actual CpG count and the absolute error are also shown.
""")
