
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./LegalBert")
model = AutoModelForSeq2SeqLM.from_pretrained("./LegalBert")

# Define a paraphrase function
def paraphrase(text, max_length=50, num_return_sequences=5, num_beams=5):
    inputs = tokenizer.encode("paraphrase: " + text, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences, num_beams=num_beams)
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases

# Streamlit frontend
st.title('Legal Document Paraphraser')
st.write('Input a legal paragraph to paraphrase:')
text = st.text_area("", "Enter text here...")
if st.button('Paraphrase'):
    paraphrased_texts = paraphrase(text)
    for i, paraphrase_text in enumerate(paraphrased_texts):
        st.write(f"Paraphrase {i+1}:")
        st.write(paraphrase_text)
