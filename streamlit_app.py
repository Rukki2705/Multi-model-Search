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
import atexit  # Import the atexit module

# Set environment variables for Huggingface and Pinecone API keys
os.environ["HUGGINGFACE_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "6d539478-7754-4b85-9a20-38960d5cc24a"

# Get API keys from environment variables
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone using the new Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'audio-search-index'

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
    
    if audio_bytes is not None and len(audio_bytes) > 0:
        audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
        return audio_array, sampling_rate, text
    else:
        return None, None, None

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

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)
        st.write(f"Upserted {len(vectors_to_upsert)} vectors.")
    else:
        st.write("No vectors to upsert.")

# Filter results by relevance
def filter_results_by_relevance(results, query_text, df):
    relevant_results = []
    query_terms = set(query_text.lower().split())

    for result in results:
        audio_id = result['id'].replace('text_', '').replace('audio_', '')
        audio_array, sampling_rate, audio_text = play_audio_and_get_text_by_id(audio_id, df)

        if audio_array is not None and audio_text is not None:
            audio_text_lower = audio_text.lower()
            if any(term in audio_text_lower for term in query_terms):
                relevant_results.append({
                    'id': result['id'],
                    'score': result['score'],
                    'audio_array': audio_array,
                    'sampling_rate': sampling_rate,
                    'text': audio_text
                })

    return relevant_results

# Search similar audios
def search_similar_audios(query_text, df, top_k=10):
    text_embedding = create_text_embeddings(query_text)
    
    search_results = index.query(vector=text_embedding, top_k=top_k)

    if 'matches' in search_results and search_results['matches']:
        matches = search_results['matches']
        st.write(f"Found {len(matches)} results for your query.")

        unique_results = {match['id']: match for match in matches}.values()
        sorted_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)

        relevant_results = filter_results_by_relevance(sorted_results, query_text, df)

        if relevant_results:
            for match in relevant_results[:5]:
                st.audio(match['audio_array'], format='audio/wav', sample_rate=match['sampling_rate'])
                st.write(f"Text: {match['text']}")
                st.write(f"Score: {match['score']}")
        else:
            st.write("No relevant matches found.")
    else:
        st.write("No results found for your query.")

# Register function to delete the index on exit
def delete_index_on_exit():
    if index_name in pc.list_indexes().names():
        st.write(f"Deleting Pinecone index '{index_name}' on exit...")
        pc.delete_index(index_name)

# Register the function with atexit
atexit.register(delete_index_on_exit)

# Streamlit UI
st.title("Audio Search Engine")

# Step 1: Text input to enter the file path
parquet_file_path = st.text_input("Enter the full path to the Parquet file containing audio data:")

if parquet_file_path:
    try:
        parquet_file_path = parquet_file_path.replace("\\", "/")
        
        if not os.path.exists(parquet_file_path):
            st.error(f"File not found at {parquet_file_path}. Please check the path and try again.")
        else:
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
