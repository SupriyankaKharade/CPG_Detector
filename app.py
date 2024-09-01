import streamlit as st
import torch
import torch.nn as nn

# Import your model and necessary functions
from your_model_file import CpGPredictor, one_hot_encode, dnaseq_to_intseq

# Load the trained model
model = CpGPredictor()
model.load_state_dict(torch.load('cpg_detector_model.pth', map_location=torch.device('cpu')))
model.eval()

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

