import streamlit as st
import requests
from PyPDF2 import PdfReader
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf


from pydub import AudioSegment
from io import BytesIO





# --- Page Configuration ---
st.set_page_config(page_title="Financial Jargon Translator", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body, .stApp {height: 100%; margin: 0; padding: 0; background: linear-gradient(135deg, #000000 75%, #0a1633 75%);}
        h1, h2, h3, label, .stMarkdown, .stTextArea label, .stSelectbox label {color: white !important;}
        .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {background-color: #181c24 !important; color: white !important;}
        div.stButton > button {
            background-color: #d7263d;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 0.6em 1.2em;
            font-size: 1.1em;
            font-weight: bold;
            transition: background 0.2s ease;
        }
        div.stButton > button:hover {background-color: #a81d2b; color: #fff;}
        .stFileUploader {background-color: #181c24 !important; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
for key, default in [
    ("text_input", ""),
    ("explanation", "")
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_classification_model():
    model = TFDistilBertForSequenceClassification.from_pretrained("./model")
    tokenizer = DistilBertTokenizer.from_pretrained("./model")
    return model, tokenizer

def classify_text(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    outputs = model(tokens)
    prediction = tf.argmax(outputs.logits, axis=1).numpy()[0]
    return prediction == 1  # True if Financial

# --- IBM Granite Functions ---
def get_access_token(api_key):
    iam_url = "https://iam.cloud.ibm.com/identity/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"apikey": api_key, "grant_type": "urn:ibm:params:oauth:grant-type:apikey"}
    response = requests.post(iam_url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get access token: {response.text}")

def query_granite(prompt_text, access_token, project_id):
    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2024-05-29"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    payload = {
        "project_id": project_id,
        "model_id": "ibm/granite-3-3-8b-instruct",
        "input": prompt_text,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "min_new_tokens": 50,
            "stop_sequences": ["\n\n"],
            "repetition_penalty": 1.1
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["results"][0]["generated_text"]
    else:
        return f"⚠️ Error {response.status_code}: {response.text}"

# --- Load the classification model ---
try:
    model, tokenizer = load_classification_model()
except Exception as e:
    st.error(f"Error loading classification model: {e}")
    model, tokenizer = None, None

# --- Load secrets ---
api_key = st.secrets["IBM_API_KEY"]
project_id = st.secrets["IBM_PROJECT_ID"]

# --- UI ---
st.markdown("<h1 style='text-align:center;'>💰 Financial Jargon Translator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a PDF or type a complex financial statement to get a simplified explanation tailored to your background.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Financial Document (PDF Optional)", type=["pdf"])

text_input = st.text_area(
    "🔍 Enter complex financial statement",
    height=100,
    key="text_input"
)

user_role = st.selectbox("🧑 Who are you?", ["Student", "Investor", "Employee"])

pdf_context = ""
if uploaded_file is not None:
    reader = PdfReader(uploaded_file)
    pdf_context = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

if st.button("Translate and Explain"):
    input_text = st.session_state.get("text_input", "")
    if not input_text.strip():
        st.warning("Please enter a statement to translate.")
    elif not model or not tokenizer:
        st.error("Classification model not loaded. Make sure your './model' folder exists and contains all necessary files.")
    else:
        try:
            is_financial = classify_text(input_text, model, tokenizer)
        except Exception as e:
            st.error(f"Error during classification: {e}")
            is_financial = False

        if not is_financial:
            st.warning("Please enter a query relevant to finance.")
        else:
            base_prompt = f"You are a financial tutor helping a {user_role.lower()} understand the following:\n"
            context_prompt = f"\nContext from uploaded document:\n{pdf_context}\n" if pdf_context else ""
            full_prompt = f"{base_prompt}{context_prompt}\nStatement:\n{input_text}\n\nExplain in simple terms."
            try:
                with st.spinner("💬 Generating explanation..."):
                    access_token = get_access_token(api_key)
                    explanation = query_granite(full_prompt, access_token, project_id)
                st.session_state.explanation = explanation.strip()
                st.success("✅ Simplified Explanation:")
                st.write(st.session_state.explanation)
            except Exception as e:
                st.error(f"Error: {e}")

if st.session_state.explanation:
    if st.button("🔍 Explain More"):
        followup_prompt = f"Explain the following even more simply for a {user_role.lower()}:\n\n{st.session_state.explanation}"
        try:
            with st.spinner("✨ Breaking it down more..."):
                access_token = get_access_token(api_key)
                deeper_explanation = query_granite(followup_prompt, access_token, project_id)
            st.success("🧠 Even Simpler Explanation:")
            st.write(deeper_explanation)
        except Exception as e:
            st.error(f"Error: {e}")


