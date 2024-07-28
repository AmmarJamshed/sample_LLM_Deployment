#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Title for the Streamlit app
st.title("NLP Model Demo with Streamlit")

# Sidebar for selecting the task
task = st.sidebar.selectbox("Choose a task", ["Text Classification", "Text Generation"])

# Function to load the model and tokenizer
def load_model(task_type):
    if task_type == "Text Classification":
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    elif task_type == "Text Generation":
        return pipeline("text-generation", model="gpt2")

# Load the appropriate pipeline based on the selected task
if task == "Text Classification":
    st.subheader("Text Classification")
    classifier = load_model(task)

    # Text input
    input_text = st.text_area("Enter the text to classify:", "")

    if st.button("Classify"):
        # Perform classification
        if input_text:
            result = classifier(input_text)
            # Display the result
            for r in result:
                st.write(f"Label: {r['label']}, Score: {r['score']:.4f}")
        else:
            st.write("Please enter some text to classify.")

elif task == "Text Generation":
    st.subheader("Text Generation")
    generator = load_model(task)

    # Text input
    input_text = st.text_area("Enter the text to start generating from:", "")

    # Number of words to generate
    num_words = st.slider("Number of words to generate", min_value=10, max_value=100, value=50)

    if st.button("Generate"):
        # Perform text generation
        if input_text:
            result = generator(input_text, max_length=num_words, num_return_sequences=1)
            # Display the result
            for r in result:
                st.write(f"Generated text: {r['generated_text']}")
        else:
            st.write("Please enter some text to start generating from.")

# Run the Streamlit app with the command: `streamlit run app.py`


# In[ ]:




