import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import plotly.express as px
import pandas as pd
import speech_recognition as sr
from huggingface_hub import hf_hub_download
from warnings import filterwarnings

# IGNORE WARNINGS
filterwarnings("ignore")

# PAGE CONFIGURATION
st.set_page_config(
    page_title="EmoAxis - Multi-label Emotion Prediction",
    layout="wide",
    page_icon="😊"
)

# CONFIGURATION 
ROBERTA_NAME = "roberta-base"
MAX_LEN = 128
DEFAULT_TOPK = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DOWNLOAD MODEL CHECKPOINT
CKPT_PATH = hf_hub_download(
    repo_id="Hidden-States/roberta-base-go-emotions",
    filename="model.pt"
)

# MODEL CLASSES

class Encoder(nn.Module):
    def __init__(self, base_encoder):
        super().__init__()
        self.encoder = base_encoder

    def forward(self, inputs: dict):
        outputs = self.encoder(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
        text_emb = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(text_emb, p=2, dim=1)

class Classifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=28):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )

    def forward(self, h):
        return self.mlp(h)

class EmoAxis(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, inputs: dict):
        text_emb = self.encoder(inputs)
        logits = self.classifier(text_emb)
        return text_emb, logits

# LOAD MODEL
@st.cache_resource(show_spinner=False)
def load_models_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)
    base = AutoModel.from_pretrained(ROBERTA_NAME)

    encoder = Encoder(base)
    classifier = Classifier()
    model = EmoAxis(encoder, classifier)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    return tokenizer, model, device

# VOICE INPUT
def voice_input():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=5, phrase_time_limit=10)

        text = r.recognize_google(audio)
        return text

    except sr.WaitTimeoutError:
        st.warning("Listening timed out.")
        return None
    except sr.UnknownValueError:
        st.warning("Could not understand speech.")
        return None
    except sr.RequestError as e:
        st.error(f"Speech recognition service error: {e}")
        return None

# APP HEADER

st.title("EmoAxis — Multi-label Emotion Prediction")
st.write("#### By: Arnab & Subinoy\n")
st.markdown("---")

with st.spinner("Loading models..."):
    try:
        tokenizer, trained_model, device = load_models_and_tokenizer()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# EMOTION LABELS
emotion_labels = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise","neutral"
]


# INPUT UI (TEXT + VOICE)

st.subheader("Input Sentence", help="You can either type a sentence or use the voice input feature to speak.")
st.info("Please correct any spelling and formatting errors to achieve the best results..")

if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""

manual_text = st.text_input(
    "Sentence",
    value=st.session_state.manual_text
)

col_voice, _ = st.columns([1, 3])

with col_voice:
    if st.button("🎙️ Speak"):
        with st.spinner("Listening..."):
            spoken_text = voice_input()
            if spoken_text:
                st.session_state.manual_text = spoken_text
                st.rerun()

# SESSION STATE SYNC
st.session_state.manual_text = manual_text

# CONTROLS
col1, col2 = st.columns(2)
topk = col1.slider("Top K", 1, 10, DEFAULT_TOPK)
threshold = col2.slider("Probability threshold", 0.0, 1.0, 0.5, 0.01)

predict_btn = st.button("Predict", type="primary")

# PREDICTION

def predict_sentence(text: str, topk: int, thr: float):
    if not text:
        return None

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt"
    )

    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        _, logits = trained_model(tokenized)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    top_idx = probs.argsort()[-topk:][::-1]
    results = [{
        "label": emotion_labels[i],
        "prob": float(probs[i]),
        "predicted": probs[i] >= thr
    } for i in top_idx]

    return results, probs

if predict_btn:
    if not manual_text.strip():
        st.warning("Please enter or speak a sentence to analyze.")
    else:
        with st.spinner("Predicting..."):
            output = predict_sentence(manual_text, topk, threshold)

            if output is None:
                st.error("Prediction failed.")
            else:
                results, probs = output

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Top Predictions")
                    for r in results:
                        st.write(f"**{r['label']}** — {r['prob'] * 100:.1f}%")

                df = pd.DataFrame({
                    "emotion": emotion_labels,
                    "prob": probs
                }).sort_values("prob", ascending=False).reset_index(drop=True)

                with col2:
                    st.subheader("All emotion probabilities")
                    st.dataframe(df.style.format({"prob": "{:.4f}"}), height=350, width=450)

                st.subheader("Top probabilities")
                top_df = df.head(10).copy()
                top_df["prob_pct"] = top_df["prob"] * 100

                col3, col4 = st.columns(2)

                with col3:
                    fig = px.bar(
                        top_df,
                        x="emotion",
                        y="prob",
                        color="prob",
                        color_continuous_scale="Viridis",
                        labels={"prob": "Probability", "emotion": "Emotion"},
                        text="prob_pct",
                        hover_data={"prob":":.2f","prob_pct": ":.2f"}
                    )
                    fig.update_traces(
                        texttemplate="%{text:.1f}%",
                        textposition="outside",
                    )
                    fig.update_layout(
                        bargap=0.2,
                        height=600,
                        width=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig)

                with col4:
                    fig1 = px.pie(
                        top_df,
                        names="emotion",
                        values="prob_pct",
                        labels={"emotion": "Emotion", "prob_pct": "Probability"},
                        hover_data={"prob_pct": ":.1f"},
                        color_discrete_sequence=px.colors.sequential.Viridis_r
                    )
                    fig1.update_traces(
                        textposition="inside",
                        texttemplate="%{label}<br>%{value:.1f}%",
                        hoverinfo="skip"
                    )

                    st.plotly_chart(fig1)
