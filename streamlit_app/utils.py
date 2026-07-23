import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
import streamlit as st
 
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from sentence_transformers import SentenceTransformer
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
 
HF_ISSUE_REPO = "shriiisstea/issue-classifier-distilbert"
HF_EMOTION_REPO = "shriiisstea/emotion-detector-distilbert"
 
RESOLUTION_MODEL_PATH = os.path.join(MODELS_DIR, "resolution_category_model.pkl")
RESOLUTION_ENCODERS_PATH = os.path.join(MODELS_DIR, "resolution_label_encoders.pkl")
RESOLUTION_TARGET_ENCODER_PATH = os.path.join(MODELS_DIR, "resolution_category_encoder.pkl")
RESOLUTION_MAPPING_PATH = os.path.join(MODELS_DIR, "resolution_mapping.pkl")
 
ESCALATION_MODEL_PATH = os.path.join(MODELS_DIR, "escalation_model.pkl")
ESCALATION_ENCODERS_PATH = os.path.join(MODELS_DIR, "escalation_encoders.pkl")
ESCALATION_TARGET_ENCODER_PATH = os.path.join(MODELS_DIR, "escalation_target_encoder.pkl")
ESCALATION_MAPPING_PATH = os.path.join(MODELS_DIR, "escalation_mapping.pkl")
 
SIMILARITY_MODEL_PATH = os.path.join(MODELS_DIR, "similarity_model.pkl")
TICKET_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "ticket_embeddings.npy")
TICKET_DATASET_PATH = os.path.join(MODELS_DIR, "ticket_dataset.csv")
 
# --- Feature sets (these genuinely differ between the two models now) -------
RESOLUTION_CATEGORICAL_COLUMNS = [
    "issue_type", "priority", "channel", "customer_segment", "platform",
    "region", "product_area", "customer_sentiment_derived", "has_attachment",
    "sla_plan",
]
RESOLUTION_NUMERIC_COLUMNS = ["reopened", "message_length", "word_count", "csat_score"]
RESOLUTION_FEATURE_COLUMNS = [
    "issue_type", "priority", "channel", "customer_segment", "platform",
    "region", "product_area", "customer_sentiment_derived", "has_attachment",
    "sla_plan", "reopened", "message_length", "word_count", "csat_score",
]
 
ESCALATION_CATEGORICAL_COLUMNS = [
    "issue_type", "priority", "channel", "customer_segment", "platform",
    "region", "product_area", "customer_sentiment_derived", "has_attachment",
]
ESCALATION_FEATURE_COLUMNS = ESCALATION_CATEGORICAL_COLUMNS  # identical, no numeric
 
# A brand-new ticket has no CSAT survey response yet (that's collected after
# resolution) — use the dataset's neutral midpoint (1-5 scale) as a default.
DEFAULT_CSAT_SCORE = 3
 
# Heuristic bridge: emotion detector (Ekman, GoEmotions-trained) -> the
# 5-class customer_sentiment_derived feature that 06/07 were actually
# trained on (see 02_sentiment_consistency_fix.py for how it was derived
# at training time — this must match that mapping exactly).
EMOTION_TO_SENTIMENT = {
    "joy": "positive",
    "neutral": "neutral",
    "surprise": "neutral",
    "sadness": "negative",
    "fear": "negative",
    "disgust": "very_negative",
    "anger": "very_negative",
}
 
EMOTION_EMOJI = {
    "joy": "😊", "neutral": "😐", "surprise": "😮",
    "sadness": "😢", "fear": "😨", "disgust": "🤢", "anger": "😠",
}

# ---------------------------------------------------------------------------
# Priority suggestion — RULE-BASED, not a trained model.
#
# Why: a diagnostic pass (crosstabs of priority against every categorical
# and numeric field in the dataset) found priority uncorrelated with
# everything available — text included. No model, ML or otherwise, can
# learn a relationship this synthetic dataset never encoded.
#
# That doesn't mean the company has no way to set priority, though — it
# means the answer here is a transparent rule engine instead of a trained
# classifier. This mirrors how a lot of real support tooling actually
# works: keyword/urgency detection + SLA tier + customer segment feeding a
# suggestion, with a human agent confirming or overriding it. It's labeled
# as a suggestion in the UI, never silently auto-applied.
# ---------------------------------------------------------------------------
URGENT_KEYWORDS = [
    "urgent", "immediately", "asap", "emergency", "right now",
    "security breach", "hacked", "compromised", "data breach",
    "lawsuit", "legal action", "unauthorized access",
]
HIGH_KEYWORDS = [
    "frustrated", "unacceptable", "very unhappy", "extremely disappointed",
    "refund", "overcharged", "not working", "broken", "failed payment",
    "cannot access", "locked out",
]

SENSITIVE_PRODUCT_AREAS = {"login_auth", "api_integration", "billing"}


def suggest_priority(ticket_text: str, sla_plan: str, customer_segment: str, product_area: str) -> str:
    text = ticket_text.lower()
    score = 0

    if any(k in text for k in URGENT_KEYWORDS):
        score += 3
    if any(k in text for k in HIGH_KEYWORDS):
        score += 2

    if sla_plan == "platinum":
        score += 2
    elif sla_plan == "gold":
        score += 1

    if customer_segment == "enterprise":
        score += 1

    if product_area in SENSITIVE_PRODUCT_AREAS:
        score += 1

    if score >= 5:
        return "urgent"
    elif score >= 3:
        return "high"
    elif score >= 1:
        return "medium"
    return "low"
 
# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading issue classification model (DistilBERT)...")
def load_issue_classifier():
    model_dir = snapshot_download(repo_id=HF_ISSUE_REPO)

    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    return tokenizer, model, label_encoder
    
 
 
@st.cache_resource(show_spinner="Loading emotion detection model (DistilBERT)...")
def load_emotion_detector():
    model_dir = snapshot_download(repo_id=HF_EMOTION_REPO)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))

    return tokenizer, model, label_encoder
    
 
 
@st.cache_resource(show_spinner="Loading resolution-time model (XGBoost)...")
def load_resolution_model():
    model = joblib.load(RESOLUTION_MODEL_PATH)
    encoders = joblib.load(RESOLUTION_ENCODERS_PATH)
    target_encoder = joblib.load(RESOLUTION_TARGET_ENCODER_PATH)
    mapping = joblib.load(RESOLUTION_MAPPING_PATH)
    return model, encoders, target_encoder, mapping
 
 
@st.cache_resource(show_spinner="Loading escalation-risk model (XGBoost)...")
def load_escalation_model():
    model = joblib.load(ESCALATION_MODEL_PATH)
    encoders = joblib.load(ESCALATION_ENCODERS_PATH)
    target_encoder = joblib.load(ESCALATION_TARGET_ENCODER_PATH)
    mapping = joblib.load(ESCALATION_MAPPING_PATH)
    return model, encoders, target_encoder, mapping
 
 
@st.cache_resource(show_spinner="Loading similar-ticket retrieval index...")
def load_similarity_index():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    nn_model = joblib.load(SIMILARITY_MODEL_PATH)
    embeddings = np.load(TICKET_EMBEDDINGS_PATH)
    ticket_df = pd.read_csv(TICKET_DATASET_PATH)  # columns: text, label
    return embedding_model, nn_model, embeddings, ticket_df
 
 
def load_all_models():
    return {
        "issue": load_issue_classifier(),
        "emotion": load_emotion_detector(),
        "resolution": load_resolution_model(),
        "escalation": load_escalation_model(),
        "similarity": load_similarity_index(),
    }
 
 
# ---------------------------------------------------------------------------
# Dropdown options, read live from the fitted encoders (union across both
# models, since resolution has one extra categorical: sla_plan)
# ---------------------------------------------------------------------------
def get_categorical_options():
    _, res_encoders, _, _ = load_resolution_model()
    _, esc_encoders, _, _ = load_escalation_model()
 
    options = {}
    for col, encoder in {**esc_encoders, **res_encoders}.items():
        if col == "has_attachment":
            options[col] = ["No", "Yes"]
        else:
            options[col] = list(encoder.classes_)
    return options
 
 
# ---------------------------------------------------------------------------
# Inference: Issue type classification (DistilBERT, top-3)
# ---------------------------------------------------------------------------
def predict_issue_type(text, top_k=3):
    tokenizer, model, label_encoder = load_issue_classifier()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=48)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    top = torch.topk(probs, k=min(top_k, probs.shape[0]))
    return [
        {"label": label_encoder.inverse_transform([idx.item()])[0], "confidence": round(score.item() * 100, 2)}
        for idx, score in zip(top.indices, top.values)
    ]
 
 
# ---------------------------------------------------------------------------
# Inference: Emotion detection (DistilBERT, Ekman 7-class) -> derived sentiment
# ---------------------------------------------------------------------------
def predict_emotion(text, top_k=3):
    tokenizer, model, label_encoder = load_emotion_detector()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    top = torch.topk(probs, k=min(top_k, probs.shape[0]))
    predictions = [
        {"label": label_encoder.inverse_transform([idx.item()])[0], "confidence": round(score.item() * 100, 2)}
        for idx, score in zip(top.indices, top.values)
    ]
    derived_sentiment = EMOTION_TO_SENTIMENT.get(predictions[0]["label"], "neutral")
    return predictions, derived_sentiment
 
 
# ---------------------------------------------------------------------------
# Safe categorical encoding with fallback for unseen values
# ---------------------------------------------------------------------------
def _safe_encode(value, encoder):
    value = str(value)
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return 0  # unseen category fallback
 
 
# ---------------------------------------------------------------------------
# Inference: Resolution-time category (Fast / Medium / Slow)
# ---------------------------------------------------------------------------
def predict_resolution_category(raw_values: dict):
    model, encoders, target_encoder, mapping = load_resolution_model()
 
    row = {}
    for col in RESOLUTION_CATEGORICAL_COLUMNS:
        row[col] = _safe_encode(raw_values[col], encoders[col])
    for col in RESOLUTION_NUMERIC_COLUMNS:
        row[col] = raw_values[col]
 
    X = pd.DataFrame([row])[RESOLUTION_FEATURE_COLUMNS]
 
    pred_idx = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    label = target_encoder.inverse_transform([pred_idx])[0]
 
    proba_by_class = {cls: round(float(p) * 100, 2) for cls, p in zip(target_encoder.classes_, pred_proba)}
    info = mapping.get(label, {})
    return {"label": label, "hours": info.get("hours", ""), "color": info.get("color", ""), "probabilities": proba_by_class}
 
 
# ---------------------------------------------------------------------------
# Inference: Escalation risk (Low / Medium / High)
# ---------------------------------------------------------------------------
def predict_escalation_risk(raw_values: dict):
    model, encoders, target_encoder, mapping = load_escalation_model()
 
    row = {col: _safe_encode(raw_values[col], encoders[col]) for col in ESCALATION_CATEGORICAL_COLUMNS}
    X = pd.DataFrame([row])[ESCALATION_FEATURE_COLUMNS]
 
    pred_idx = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    label = target_encoder.inverse_transform([pred_idx])[0]
 
    proba_by_class = {cls: round(float(p) * 100, 2) for cls, p in zip(target_encoder.classes_, pred_proba)}
    info = mapping.get(label, {})
    return {"label": label, "color": info.get("color", ""), "description": info.get("description", ""), "probabilities": proba_by_class}
 
 
# ---------------------------------------------------------------------------
# Inference: Top-N similar historical tickets (original text+label corpus)
# ---------------------------------------------------------------------------
def find_similar_tickets(text, top_n=3, similarity_threshold=0.0):
    embedding_model, nn_model, embeddings, ticket_df = load_similarity_index()
 
    query_embedding = embedding_model.encode([text], convert_to_numpy=True)
    n_neighbors = min(nn_model.n_neighbors, len(ticket_df))
    distances, indices = nn_model.kneighbors(query_embedding, n_neighbors=n_neighbors)
 
    similarities = (1 - distances[0]) * 100
    results = ticket_df.iloc[indices[0]].copy()
    results["similarity"] = similarities
    results = results[results["similarity"] >= similarity_threshold]
    results = results.sort_values("similarity", ascending=False).head(top_n)
 
    return results[["text", "label", "similarity"]].to_dict(orient="records")
 
 
# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline(ticket_text: str, form_inputs: dict):
    """
    form_inputs must contain: priority, channel, customer_segment, platform,
    region, product_area, has_attachment (0/1), sla_plan, reopened (0/1)
    """
    issue_predictions = predict_issue_type(ticket_text, top_k=3)
    top_issue_type = issue_predictions[0]["label"]
 
    emotion_predictions, derived_sentiment = predict_emotion(ticket_text, top_k=3)
 
    message_length = len(ticket_text)
    word_count = len(ticket_text.split())
 
    shared_categorical = {
        "issue_type": top_issue_type,
        "priority": form_inputs["priority"],
        "channel": form_inputs["channel"],
        "customer_segment": form_inputs["customer_segment"],
        "platform": form_inputs["platform"],
        "region": form_inputs["region"],
        "product_area": form_inputs["product_area"],
        "customer_sentiment_derived": derived_sentiment,
        "has_attachment": form_inputs["has_attachment"],
    }
 
    resolution_row = {
        **shared_categorical,
        "sla_plan": form_inputs["sla_plan"],
        "reopened": form_inputs.get("reopened", 0),
        "message_length": message_length,
        "word_count": word_count,
        "csat_score": DEFAULT_CSAT_SCORE,
    }
 
    resolution = predict_resolution_category(resolution_row)
    escalation = predict_escalation_risk(shared_categorical)
    similar_tickets = find_similar_tickets(ticket_text, top_n=3, similarity_threshold=0.0)
 
    return {
        "issue_predictions": issue_predictions,
        "top_issue_type": top_issue_type,
        "emotion_predictions": emotion_predictions,
        "derived_sentiment": derived_sentiment,
        "resolution": resolution,
        "escalation": escalation,
        "similar_tickets": similar_tickets,
        "tabular_row": resolution_row,
    }
 