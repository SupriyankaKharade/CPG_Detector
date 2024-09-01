import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import necessary functions
from functools import partial

# Define helper functions
def count_cpgs(seq: str) -> int:
    cgs = 0
    for i in range(0, len(seq) - 1):
        dimer = seq[i:i+2]
        if dimer == "CG":
            cgs += 1
    return cgs

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(5))}
int2dna = {i: a for a, i in zip(alphabet, range(5))}
intseq_to_dnaseq = partial(map, int2dna.get)
dnaseq_to_intseq = partial(map, dna2int.get)

def one_hot_encode(sequence, num_classes=5):
    sequence_tensor = torch.tensor(sequence, dtype=torch.long)
    one_hot = F.one_hot(sequence_tensor, num_classes=num_classes)
    return one_hot.float()

# Define the model
class CpGPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, num_layers=2, output_size=1):
        super(CpGPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        logits = self.classifier(last_output)
        return logits

# Load the trained model
@st.cache_resource
def load_model():
    model = CpGPredictor()
    model.load_state_dict(torch.load('cpg_detector_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

st.title('CpG Detector')

sequence = st.text_input('Enter DNA sequence (use N, A, C, G, T only):')

if st.button('Predict'):
    if sequence and all(c in 'NACGT' for c in sequence):
        int_sequence = list(dnaseq_to_intseq(sequence))
        one_hot = one_hot_encode(int_sequence).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(one_hot).item()
        
        st.write(f'Predicted CpG count: {prediction:.2f}')
        st.write(f'Actual CpG count: {count_cpgs(sequence)}')
    else:
        st.write('Invalid sequence. Please use only N, A, C, G, T.')
