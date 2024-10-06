import streamlit as st
import torch
import clip  # CLIP model from OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone, ServerlessSpec, PineconeException
import textwrap
import os
import time
import nltk
from nltk.corpus import stopwords
import wave
from transformers import ClapProcessor, ClapModel, CLIPProcessor, CLIPModel
from io import BytesIO
import pandas as pd
import numpy as np
import re
from PIL import Image
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import tempfile
from datetime import datetime, timedelta

# Custom CSS to style the app
st.markdown("""
    <style>
    /* Style the green buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px;
        cursor: pointer;
        border-radius: 12px;
        width: 150px; /* Fixed width for all buttons */
        height: 60px; /* Fixed height for all buttons */
    }

    /* Center the buttons */
    .stButton {
        display: flex;
        justify-content: center;
    }

    /* Center the main heading text */
    .centered-text {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit title and description
st.title("🌿 Working with Vector Databases and Performing Semantic Searches")

# Initialize session state to keep track of which page is active
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Function to switch page
def switch_page(page_name):
    st.session_state.page = page_name

# Home Page
if st.session_state.page == "home":
    # Center-align the instruction text
    st.markdown('<div class="centered-text">Choose the type of search you\'d like to perform:</div>', unsafe_allow_html=True)

    # Creating columns for buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🖼️ Image Search"):
            switch_page("image")
    with col2:
        if st.button("✍️ Text Search"):
            switch_page("text")
    with col3:
        if st.button("🎥 Video Search"):
            switch_page("video")
    with col4:
        if st.button("🎵 Audio Search"):
            switch_page("audio")


# ------------------- Image Search Functionality -------------------
if st.session_state.page == "image":
    st.title("🖼️ Image Search with CLIP and Pinecone")

    # Add a "Home" button
    if st.button("🏠 Home"):
        switch_page("home")
        st.stop()

    # Sidebar for Pinecone options and Models used
    st.sidebar.title("⚙️ Options")

    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options, key="pinecone_image")

    if selected_pinecone_option == "API Key":
        st.sidebar.write("Current API Key: bbc775b8-dc2a-4136-b80f-d1bac425405b")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: interactive-clip-index")

    model_options = ["CLIP ViT-B/32", "Sentence-BERT"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options, key="model_image")

    if selected_model_option == "CLIP ViT-B/32":
        st.sidebar.write("Model: CLIP ViT-B/32 by OpenAI")
    elif selected_model_option == "Sentence-BERT":
        st.sidebar.write("Model: Sentence-BERT used for text embeddings")

    # Step 1: Initialize Pinecone client
    if 'pinecone_initialized' not in st.session_state:
        try:
            pc = Pinecone(
                api_key='6d539478-7754-4b85-9a20-38960d5cc24a',  # Use your Pinecone API key
                environment='us-east-1'  # Replace with your Pinecone environment
            )
            st.session_state.pc = pc
            st.session_state.pinecone_initialized = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Interactive step: Button to create Pinecone index
    index_name = "interactive-clip-index"
    if st.button("🛠️ Create Index"):
        if 'index_created' not in st.session_state:
            try:
                existing_indexes = st.session_state.pc.list_indexes()

                if index_name in existing_indexes:
                    st.write(f"Index '{index_name}' already exists. Connecting to the existing index...")
                else:
                    st.write(f"Creating a new index '{index_name}'...")
                    st.session_state.pc.create_index(
                        name=index_name,
                        dimension=512,  # CLIP's ViT-B/32 outputs 512-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                st.session_state.index_created = True
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    if 'index' not in st.session_state and 'index_created' in st.session_state:
        st.session_state.index = st.session_state.pc.Index(index_name)

    # Step 2: Select data type (currently only supporting images)
    data_type = st.selectbox("📂 Select Data Type", ["Images"], key="data_type_image")

    # Step 3: Upload files
    uploaded_files = st.file_uploader("📤 Choose files", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} files uploaded successfully.")

    # Step 4: Preview Data
    if st.button("🔍 Preview Uploaded Files"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Preview of {uploaded_file.name}", use_column_width=True)
        else:
            st.write("No files uploaded yet.")

    # Step 5: Convert to Embedding
    if st.button("🧠 Convert to Embedding"):
        if uploaded_files:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load CLIP model and preprocessing
            if 'model' not in st.session_state:
                model, preprocess = clip.load("ViT-B/32", device=device)
                st.session_state.model = model
                st.session_state.preprocess = preprocess
            else:
                model = st.session_state.model
                preprocess = st.session_state.preprocess

            image_directory = "images"
            os.makedirs(image_directory, exist_ok=True)

            for uploaded_file in uploaded_files:
                image_path = os.path.join(image_directory, uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_embedding = model.encode_image(image).cpu().numpy().flatten()

                if 'image_embeddings' not in st.session_state:
                    st.session_state.image_embeddings = []
                st.session_state.image_embeddings.append({"filename": uploaded_file.name, "embedding": image_embedding})

            st.write("Images converted to embeddings successfully.")
        else:
            st.write("No files uploaded yet.")

    # Step 6: Store embeddings
    if st.button("💾 Store Embeddings"):
        if 'image_embeddings' in st.session_state:
            index = st.session_state.index
            for image_data in st.session_state.image_embeddings:
                index.upsert(
                    vectors=[
                        {
                            "id": image_data['filename'],
                            "values": image_data['embedding'].tolist(),
                            "metadata": {"filename": image_data['filename']}
                        }
                    ]
                )
            st.write("Embeddings stored in Pinecone successfully.")
        else:
            st.write("No embeddings to store. Please convert the images to embeddings first.")

    # Step 7: Semantic Text Search
    text_query = st.text_input("Enter your text query:", key="text_query_image")

    similarity_threshold = 0.25

    if st.button("🔎 Search"):
        if text_query:
            text_tokenized = clip.tokenize([text_query]).to(device)
            with torch.no_grad():
                text_embedding = model.encode_text(text_tokenized).cpu().numpy().flatten()

            query_results = st.session_state.index.query(
                vector=text_embedding.tolist(),
                top_k=2,  # Return top 2 results
                include_metadata=True
            )

            results_found = False
            if query_results['matches']:
                for result in query_results['matches']:
                    if result['score'] >= similarity_threshold:
                        results_found = True
                        top_result_filename = result['metadata']['filename']
                        top_result_image_path = os.path.join("images", top_result_filename)
                        top_result_image = Image.open(top_result_image_path)

                        st.image(top_result_image, caption=f"Filename: {top_result_filename} - Score: {result['score']}", use_column_width=True)

                if not results_found:
                    st.write("No results found above the similarity threshold.")

# ------------------- Text Search Functionality -------------------
if st.session_state.page == "text":
    st.title("✍️ Text Search with Sentence-BERT and Pinecone")

    # Add a "Home" button
    if st.button("🏠 Home"):
        switch_page("home")
        st.stop()

    # Pinecone Sidebar Options
    st.sidebar.title("⚙️ Options")
    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options, key="pinecone_text")

    if selected_pinecone_option == "API Key":
        st.sidebar.write("Current API Key: bbc775b8-dc2a-4136-b80f-d1bac425405b")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: sentence-transformers-pdf-index")

    # Step 1: Initialize Pinecone client for text
    if 'pinecone_initialized_text' not in st.session_state:
        try:
            pc = Pinecone(
                api_key='6d539478-7754-4b85-9a20-38960d5cc24a',
                environment='us-east-1'
            )
            st.session_state.pc_text = pc
            st.session_state.pinecone_initialized_text = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Button to create the index for text search
    index_name = "sentence-transformers-pdf-index"
    if st.button("🛠️ Create Text Index"):
        if 'index_created_text' not in st.session_state:
            try:
                existing_indexes = st.session_state.pc_text.list_indexes()

                if index_name in existing_indexes:
                    st.write(f"Index '{index_name}' already exists. Connecting to the existing index...")
                else:
                    st.write(f"Creating a new index '{index_name}'...")
                    st.session_state.pc_text.create_index(
                        name=index_name,
                        dimension=384,  # Sentence-BERT outputs 384-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                st.session_state.index_created_text = True
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    # Connect to the index
    if 'index_text' not in st.session_state and 'index_created_text' in st.session_state:
        st.session_state.index_text = st.session_state.pc_text.Index(index_name)

    # Step 2: Upload the PDF file
    uploaded_pdf = st.file_uploader("📤 Choose a PDF file", type="pdf", key="pdf_upload_text")

    if uploaded_pdf:
        st.write(f"File uploaded: {uploaded_pdf.name}")

    # Step 3: Convert PDF to embeddings and store in Pinecone
    if st.button("🧠 Convert PDF to Embedding"):
        if uploaded_pdf:
            reader = PdfReader(uploaded_pdf)
            pages = [page.extract_text() for page in reader.pages]
            model = SentenceTransformer('all-MiniLM-L6-v2')

            for idx, page_text in enumerate(pages):
                page_embedding = model.encode(page_text)
                st.session_state.index_text.upsert(
                    vectors=[
                        {
                            "id": f"page_{idx}",
                            "values": page_embedding.tolist(),
                            "metadata": {"page_content": page_text}
                        }
                    ]
                )
            st.write("PDF pages converted to embeddings and stored.")
        else:
            st.write("Please upload a PDF file first.")

    # Step 4: Perform text-based search
    text_query = st.text_input("Enter your search query:", key="text_query_pdf")

    similarity_threshold = 0.25

    if st.button("🔎 Search PDF"):
        if text_query:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            query_embedding = model.encode(text_query)

            query_results = st.session_state.index_text.query(
                vector=query_embedding.tolist(),
                top_k=3,  # Return top results
                include_metadata=True
            )

            results_found = False
            if query_results['matches']:
                for result in query_results['matches']:
                    if result['score'] >= similarity_threshold:
                        results_found = True
                        formatted_content = result['metadata']['page_content'].replace("\n", " ")
                        wrapped_content = textwrap.fill(formatted_content, width=80)
                        st.write(f"Matched Page Content:\n{'-' * 40}\n{wrapped_content}\n{'-' * 40}")
                        st.write(f"Score: {result['score']}\n")

                if not results_found:
                    st.write("No results found above the similarity threshold.")


#----------------------------Audio Search--------------------------

if st.session_state.page == "audio":
    st.title("✍️ Audio Search with CLAP and Pinecone")

    # Add a "Home" button
    if st.button("🏠 Home"):
        switch_page("home")
        st.stop()

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    st.sidebar.title("⚙️ Options")

    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options, key="pinecone_audio")

    if selected_pinecone_option == "API Key":
        st.sidebar.write("Current API Key: 6d539478-7754-4b85-9a20-38960d5cc24a")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: audio-search-index")

    model_options = ["CLAP"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options, key="model_audio")

    if selected_model_option == "CLAP":
        st.sidebar.write("Model: CLAP by LAION")

    # Step 1: Initialize Pinecone client
    if 'pinecone_initialized' not in st.session_state:
        try:
            pc = Pinecone(api_key="6d539478-7754-4b85-9a20-38960d5cc24a", environment='us-east-1')
            st.session_state.pc = pc
            st.session_state.pinecone_initialized = True
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Set the index name
    index_name = "audio-search-index"

    # Track session start time
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()

    # Function to create index
    def create_index():
        if 'index_created' not in st.session_state:
            try:
                existing_indexes = st.session_state.pc.list_indexes()

                if index_name in existing_indexes:
                    st.write(f"Index '{index_name}' already exists.")
                else:
                    st.write(f"Creating index '{index_name}'...")
                    st.session_state.pc.create_index(
                        name=index_name,
                        dimension=512,  # CLAP audio/text outputs 512-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                st.session_state.index_created = True
                st.success(f"Index '{index_name}' created successfully.")
            except Exception as e:
                st.error(f"Error creating index: {str(e)}")

    # Create index button
    if st.button("🛠️ Create Index"):
        create_index()

    # Ensure the index object is stored in session state after creation
    if 'index' not in st.session_state and 'index_created' in st.session_state:
        st.session_state.index = st.session_state.pc.Index(index_name)

    # **New**: File uploader for Parquet file
    parquet_file = st.file_uploader("Upload the Parquet file containing audio data", type=["parquet"], key="parquet_audio")

    # Define utility functions
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    def normalize_embeddings(embeddings):
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / (norm + 1e-8)

    def extract_audio_array_from_bytes(audio_bytes):
        with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
            sampling_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            audio_frames = wav_file.readframes(frames)
        audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        return audio_array, sampling_rate

    def create_audio_embeddings(audio_array, sampling_rate):
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        inputs = processor(audios=audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_audio_features(**inputs).cpu().numpy()
        return normalize_embeddings(embeddings).squeeze().tolist()

    def create_text_embeddings(text):
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        preprocessed_text = preprocess_text(text)
        inputs = processor(text=preprocessed_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs).cpu().numpy()
        return normalize_embeddings(embeddings).squeeze().tolist()

    def store_audio_and_text_embeddings(audio_embedding, text_embedding, audio_id, metadata):
        index = st.session_state.pc.Index(index_name)  # Use st.session_state.pc to get Pinecone client
        index.upsert(vectors=[
            {"id": f"audio_{audio_id}", "values": audio_embedding, "metadata": metadata},
            {"id": f"text_{audio_id}", "values": text_embedding, "metadata": metadata}
        ])

    def play_audio_and_get_text_by_id(audio_id, df):
        row = df[df['line_id'] == audio_id].iloc[0]
        audio_bytes = row['audio']['bytes']
        text = row['text']

        if audio_bytes is not None and len(audio_bytes) > 0:
            audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
            return audio_array, sampling_rate, text
        else:
            return None, None, None    

    def display_first_5_audios(df):
        st.write("Displaying first 5 audio samples and associated text:")
        for idx, row in df.iterrows():
            if idx >= 5:
                break  # Limit to the first 5 audios
            audio_id = row['line_id']
            audio_bytes = row['audio']['bytes']
            text_associated = row['text']

            if audio_bytes:
                audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
                st.audio(audio_array, format="audio/wav", sample_rate=sampling_rate)
                st.write(f"Text associated with audio {audio_id}: {text_associated}")
            else:
                st.write(f"Audio for row {audio_id} is not available.")

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

    def search_similar_audios(query_text, df, top_k=10):
        text_embedding = create_text_embeddings(query_text)

        index = st.session_state.pc.Index(index_name)  # Use st.session_state.pc to get Pinecone client
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

    # **Updated Handling** Parquet file upload and embedding creation
    if parquet_file is not None:
        try:
            df = pd.read_parquet(parquet_file)
            st.write("Dataset loaded successfully.")

            if st.button("🔍 Preview Data"):
                display_first_5_audios(df)

            if st.button("Create and Store Audio Embeddings"):
                for idx, row in df.iterrows():
                    audio_id = row['line_id']
                    audio_bytes = row['audio']['bytes']
                    text_associated = row['text']
                    if audio_bytes:
                        audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
                        audio_embeddings = create_audio_embeddings(audio_array, sampling_rate)
                        text_embeddings = create_text_embeddings(text_associated)
                        metadata = {"text": text_associated}
                        store_audio_and_text_embeddings(audio_embeddings, text_embeddings, audio_id, metadata)
                st.write("All audio and text embeddings stored successfully.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Query input for searching similar audio
    query_text = st.text_input("Enter a query to search similar audio:", key="query_audio")
    if query_text and st.button("Search Similar Audio"):
        search_similar_audios(query_text, df)





#---------------------------------Video Search-----------------------------------------------

def get_pinecone_client():
    if 'pinecone_initialized' not in st.session_state:
        try:
            pc = Pinecone(api_key="6d539478-7754-4b85-9a20-38960d5cc24a", environment='us-east-1')
            st.session_state.pc = pc
            st.session_state.pinecone_initialized = True
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return None
    return st.session_state.pc

if st.session_state.page == "video":
    st.title("✍️ Video Search with CLIP and Pinecone")

    # Add a "Home" button
    if st.button("🏠 Home"):
        switch_page("home")
        st.stop()

    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options)

    if selected_pinecone_option == "API Key":
        st.sidebar.write("Current API Key: 6d539478-7754-4b85-9a20-38960d5cc24a")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: video-search-index")

    model_options = ["CLIP"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options)

    if selected_model_option == "CLIP":
        st.sidebar.write("Model: CLIP by OpenAI")

    # Step 1: Initialize Pinecone client
    pc = get_pinecone_client()
    if pc is None:
        st.stop()

    # Step 2: Interactive step to create Pinecone index
    index_name = "video-search-index"
    
    if st.button("🛠️ Create Index"):
        if 'index_created' not in st.session_state or not st.session_state.index_created:
            try:
                existing_indexes = pc.list_indexes()
                st.write(f"Existing indexes: {existing_indexes}")  # Debugging info

                if index_name in existing_indexes:
                    st.write(f"Index '{index_name}' already exists. Connecting to the existing index...")
                else:
                    st.write(f"Creating a new index '{index_name}'...")
                    pc.create_index(
                        name=index_name,
                        dimension=512,  # CLIP audio/text outputs 512-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                    st.session_state.index_created = True
                    st.write(f"Index '{index_name}' created successfully.")
            except Exception as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    if 'index' not in st.session_state and 'index_created' in st.session_state and st.session_state.index_created:
        st.session_state.index = pc.Index(index_name)

    video_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")    

    def get_single_frame_from_video(video_capture, time_sec):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        success, frame = video_capture.read()
        if success:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    # Function to divide video into intervals and extract frames
    def get_frames_from_video_by_interval(video_path, interval_sec=10):
        frames = []
        video_capture = cv2.VideoCapture(video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_duration_sec = total_frames / fps

        for time_sec in np.arange(0, total_duration_sec, interval_sec):
            frame = get_single_frame_from_video(video_capture, time_sec)
            if frame is not None:
                frames.append(frame)
        
        video_capture.release()
        return frames

    # Function to create image embeddings using CLIP
    def get_image_embedding(image):
        inputs = clip_processor(images=image, return_tensors="pt")
        image_embeddings = video_model.get_image_features(**inputs)
        
        return list(map(float, image_embeddings[0].detach().numpy().astype(np.float32)))

    # Function to create text embedding using CLIP
    def get_text_embedding(text):
        inputs = clip_processor(text=[text], return_tensors="pt")
        text_embedding = video_model.get_text_features(**inputs)
        
        return list(map(float, text_embedding[0].detach().numpy().astype(np.float32)))

    # Main process to extract frames, create embeddings, and store in Pinecone
    def process_video_for_embedding(video_path, interval_sec=10):
        frames = get_frames_from_video_by_interval(video_path, interval_sec)
        
        image_embeddings = []
        image_ids = []
        for i, frame in enumerate(frames):
            embedding = get_image_embedding(frame)
            image_embeddings.append(embedding)
            image_ids.append(str(i))

        pinecone_vectors = [
            (image_ids[i], image_embeddings[i])
            for i in range(len(image_embeddings))
        ]
        
        index = st.session_state.pc.Index(index_name)
        index.upsert(vectors=pinecone_vectors)
        st.success(f"Inserted {len(pinecone_vectors)} vectors into Pinecone.")
        
        return pinecone_vectors

    # Function to search for similar video frames based on text query
    def search_video_by_text(query_text):
        query_embedding = get_text_embedding(query_text)
        
        index = st.session_state.pc.Index(index_name)
        result = index.query(vector=[query_embedding], top_k=5)
        
        if 'matches' in result and len(result['matches']) > 0:
            closest_match = result['matches'][0]
            frame_id = closest_match['id']
            similarity_score = closest_match['score']
            
            return frame_id, similarity_score
        else:
            return None, None  # No matching results found

    # Use moviepy to extract and save a video segment
    def play_video_segment(video_path, frame_id, interval_sec=10, segment_duration=5):
        if frame_id is None:
            st.error("No frame ID provided. Cannot play video.")
            return None

        # Calculate the time (in seconds) of the matching frame
        frame_time_sec = int(frame_id) * interval_sec
        start_time_sec = max(frame_time_sec - segment_duration // 2, 0)
        end_time_sec = start_time_sec + segment_duration

        # Create a temporary file to save the video segment
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_path = temp_video_file.name

        # Use moviepy's ffmpeg_extract_subclip to extract the segment
        ffmpeg_extract_subclip(video_path, start_time_sec, end_time_sec, targetname=temp_video_path)

        # Check if the video segment has been created
        if os.path.exists(temp_video_path):
            return temp_video_path
        else:
            st.error("Failed to create video segment.")
            return None

    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            temp_video_path = temp_video_file.name

        # Display the uploaded video
        st.video(temp_video_path)

        # Button to process the video and create embeddings
        if st.button("Process Video and Create Embeddings"):
            process_video_for_embedding(temp_video_path)

        # Input field for query and button to search for similar video frames
        query_text = st.text_input("Enter a query to search for similar video frames:")

        if query_text and st.button("Search for Similar Video Frames"):
            # Perform search and retrieve closest matching frame
            frame_id, score = search_video_by_text(query_text)

            if frame_id is not None:
                st.write(f"Closest frame ID: {frame_id} with similarity score: {score}")

                # Play the matching video segment
                segment_video_path = play_video_segment(temp_video_path, frame_id)

                if segment_video_path:
                    st.video(segment_video_path)
                else:
                    st.error("No matching video segment found for the query.")
