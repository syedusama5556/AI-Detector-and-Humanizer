import sys
import gradio as gr
from gramformer import Gramformer
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import nltk
import spacy
from nltk.corpus import wordnet
import subprocess
import os

# Download NLTK data
nltk.download("punkt", download_dir="./nltk_data")
nltk.download("stopwords", download_dir="./nltk_data")
nltk.download("wordnet", download_dir="./nltk_data")
nltk.data.path.append("./nltk_data")

# Download spaCy model if not already installed
if not os.path.exists("./spacy_model"):
    os.makedirs("./spacy_model")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "spacy",
            "download",
            "en_core_web_sm",
            "--target",
            "./spacy_model",
        ]
    )
nlp = spacy.load("./spacy_model/en_core_web_sm/en_core_web_sm-3.5.0")

# Check for GPU and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to download and save models
def download_and_save_model(model_name, model_class, tokenizer_class):
    model_dir = f"./{model_name.replace('/', '_')}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        tokenizer = tokenizer_class.from_pretrained(model_name)
        model = model_class.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
    return tokenizer_class.from_pretrained(model_dir), model_class.from_pretrained(
        model_dir
    )


# Load AI Detector model and tokenizer
detector_tokenizer, detector_model = download_and_save_model(
    "Varun53/openai-roberta-large-AI-detection",
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
detector_model = detector_model.to(device)

# Load Paraphrase model and tokenizer for humanizing text
paraphrase_tokenizer, paraphrase_model = download_and_save_model(
    "SRDdev/Paraphrase", T5ForConditionalGeneration, T5Tokenizer
)
paraphrase_model = paraphrase_model.to(device)


# Load Grammar Correction Model
grammar_tokenizer, grammar_model = download_and_save_model(
    "prithivida/grammar_error_correcter_v1", T5ForConditionalGeneration, T5Tokenizer
)
grammar_model = grammar_model.to(device)

# Initialize Gramformer with the pre-loaded models
use_gpu = True if torch.cuda.is_available() else False
gramformer = Gramformer(grammar_tokenizer, grammar_model, use_gpu=use_gpu)


humanized_text = "I'm thrilled have come across your project, which aims us introduce and simplify an concept of business models for businesses of all sizes. Developing a business model that effectively balances profit motives with cost management is crucial in today's competitive landscape. I am confident that my expertise and experience in this field can add significant value to your project.You've highlighted the importance of a straightforward framework for businesses to create effective business models. This involves understanding the quantitative business model, which is all about revenues minus costs and expenses equaling profits. Your emphasis on a three-step process that takes into account all participants' profit motives is both insightful and necessary for fostering success."

# Step 3: Correct grammar using Gramformer with chunking
def correct_grammar_with_chunks(text):
    sentences = nltk.sent_tokenize(text)
    corrected_text = []

    for sentence in sentences:
        # Get the corrected output for each sentence
        corrected_sentence = gramformer.correct(sentence)
        # Only take the first correction if there are multiple candidates
        if corrected_sentence:
            corrected_text.append(corrected_sentence[0])  # Append the first correction

    # Join all corrected sentences into a single string
    return ' '.join(corrected_text)

# Correct grammar in the humanized text
corrected_text = correct_grammar_with_chunks(humanized_text)
print("corrected_text:", corrected_text)
