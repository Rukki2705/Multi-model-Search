import streamlit as st

import streamlit as st
import pandas as pd
import wave
import numpy as np
import torch
import nltk
from transformers import ClapProcessor, ClapModel
from nltk.corpus import stopwords
from pinecone import Pinecone, ServerlessSpec
import re
from io import BytesIO
import os

# Set environment variables for Huggingface and Pinecone API keys
os.environ["HUGGINGFACE_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "6d539478-7754-4b85-9a20-38960d5cc24a"

# Get API keys from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'audio'

# Check if the index exists before creating it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    st.write(f"Index '{index_name}' created.")
else:
    st.write(f"Index '{index_name}' already exists.")

# Connect to the Pinecone index
index = pc.Index(index_name)

# Load CLAP processor and model
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()

# Download and initialize stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocess the text input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Normalize embeddings
def normalize_embeddings(embeddings):
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / (norm + 1e-8)

# Extract audio array from bytes
def extract_audio_array_from_bytes(audio_bytes):
    with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
        sampling_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        audio_frames = wav_file.readframes(frames)
    audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
    return audio_array, sampling_rate

# Create audio embeddings
def create_audio_embeddings(audio_array, sampling_rate):
    inputs = processor(audios=audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_audio_features(**inputs).cpu().numpy()
    return normalize_embeddings(embeddings).squeeze().tolist()

# Create text embeddings
def create_text_embeddings(text):
    preprocessed_text = preprocess_text(text)
    inputs = processor(text=preprocessed_text, return_tensors="pt", padding=True)
    with torch.no_grad():
        embeddings = model.get_text_features(**inputs).cpu().numpy()
    return normalize_embeddings(embeddings).squeeze().tolist()

# Play audio and get text
def play_audio_and_get_text_by_id(audio_id, df):
    row = df[df['line_id'] == audio_id].iloc[0]
    audio_bytes = row['audio']['bytes']
    text = row['text']
    audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
    return audio_array, sampling_rate, text

# Upsert unique embeddings into Pinecone
def upsert_unique_embeddings(df):
    unique_ids = set()
    vectors_to_upsert = []

    for idx, row in df.iterrows():
        audio_id = row['line_id']
        if audio_id in unique_ids:
            continue
        unique_ids.add(audio_id)

        audio_bytes = row['audio']['bytes']
        transcript = row['text']

        if audio_bytes and transcript:
            audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
            audio_embeddings = create_audio_embeddings(audio_array, sampling_rate)
            text_embeddings = create_text_embeddings(transcript)

            vectors_to_upsert.append({
                'id': f"audio_{audio_id}",
                'values': audio_embeddings,
                'metadata': {'sampling_rate': sampling_rate}
            })

            vectors_to_upsert.append({
                'id': f"text_{audio_id}",
                'values': text_embeddings,
                'metadata': {'text': transcript}
            })

    # Only upsert if we have vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        st.write(f"Upserted {len(vectors_to_upsert)} vectors.")
    else:
        st.write("No vectors to upsert.")

# Search and display similar audios
def search_similar_audios(query_text, df, top_k=5):
    text_embedding = create_text_embeddings(query_text)
    search_results = index.query(vector=text_embedding, top_k=top_k)

    if 'matches' in search_results:
        matches = search_results['matches']
        for match in matches:
            vector_id = match['id']
            audio_id = vector_id.replace('text_', '').replace('audio_', '')
            audio_array, sampling_rate, transcript = play_audio_and_get_text_by_id(audio_id, df)
            
            # Provide sample rate explicitly when playing the audio
            st.audio(audio_array, format='audio/wav', sample_rate=sampling_rate)
            st.write(f"Text: {transcript}")
            st.write(f"Score: {match['score']}")

# Streamlit UI
st.title("Audio Search Engine")

# Step 1: File input (path provided)
parquet_file_path = st.text_input("Enter the path to the Parquet file containing audio data:")

if parquet_file_path:
    try:
        # Load the dataset
        df = pd.read_parquet(parquet_file_path)
        st.write("Dataset loaded successfully.")

        # Step 2: Upsert embeddings to Pinecone
        if st.button("Create and Store Embeddings"):
            st.write("Processing embeddings and upserting to Pinecone...")
            upsert_unique_embeddings(df)
            st.write("Embeddings stored in Pinecone successfully.")

        # Step 3: Query input
        query_text = st.text_input("Enter a query to search for similar audio:")

        if query_text:
            st.write(f"Searching for: {query_text}")
            search_similar_audios(query_text, df)

    except Exception as e:
        st.error(f"Error loading file: {e}")

