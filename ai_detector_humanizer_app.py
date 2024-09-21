import sys
import gradio as gr
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


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def replace_with_synonyms(text):
    doc = nlp(text)
    processed_text = []
    for token in doc:
        synonyms = get_synonyms(token.text.lower())
        if synonyms and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}:
            replacement = synonyms[0]
            if token.is_title:
                replacement = replacement.capitalize()
            processed_text.append(replacement)
        else:
            processed_text.append(token.text)
    return " ".join(processed_text)


def detect_ai_generated(text):
    inputs = detector_tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        logits = detector_model(**inputs).logits
        probs = logits.softmax(dim=-1)
    ai_probability = probs[0][0].item() * 100
    return f"AI-Generated Probability: {ai_probability:.2f}%"


def humanize_text(AI_text):
    text_with_synonyms = replace_with_synonyms(AI_text)
    paragraphs = text_with_synonyms.split("\n")
    paraphrased_paragraphs = []
    for paragraph in paragraphs:
        if paragraph.strip():
            inputs = paraphrase_tokenizer(
                paragraph, return_tensors="pt", max_length=512, truncation=True
            ).to(device)
            paraphrased_ids = paraphrase_model.generate(
                inputs["input_ids"],
                max_length=inputs["input_ids"].shape[-1] + 20,
                num_beams=4,
                early_stopping=True,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
            )
            paraphrased_text = paraphrase_tokenizer.decode(
                paraphrased_ids[0], skip_special_tokens=True
            )
            paraphrased_paragraphs.append(paraphrased_text)
    humanized_text = "\n\n".join(paraphrased_paragraphs)
    detection_result = detect_ai_generated(humanized_text)

    return humanized_text, detection_result


# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Detector and Humanizer")

    with gr.Tab("AI Detector"):
        with gr.Row():
            detector_input = gr.Textbox(label="Input Text", lines=5)
            detector_output = gr.Textbox(label="AI Detection Result")
        detector_button = gr.Button("Detect AI")

    with gr.Tab("AI Humanizer"):
        with gr.Row():
            humanizer_input = gr.Textbox(label="Input Text", lines=5)
            humanizer_output = gr.Textbox(label="Humanized Text", lines=5)
        detector_output_after = gr.Textbox(
            label="AI Detection Result After Humanization"
        )

        humanizer_button = gr.Button("Humanize Text")

    detector_button.click(
        detect_ai_generated, inputs=detector_input, outputs=detector_output
    )
    humanizer_button.click(
        humanize_text,
        inputs=humanizer_input,
        outputs=[humanizer_output, detector_output_after],
    )

if __name__ == "__main__":
    demo.launch()
