import streamlit as st
import os
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torch
from chromadb import PersistentClient
from chromadb.config import Settings
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import tempfile
import json
from typing import List, Dict, Any

# Set environment variables before importing torch
os.environ['TORCH_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'torch')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')

# Global variables for model and processor
_model = None
_processor = None

# Initialize session state for settings
if 'last_query' not in st.session_state:
    st.session_state.last_query = ''
if 'num_results' not in st.session_state:
    st.session_state.num_results = 5
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 32
if 'embeddings_dir' not in st.session_state:
    st.session_state.embeddings_dir = os.path.join(os.getcwd(), 'chromadb')


def select_directory():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    directory = filedialog.askdirectory()
    root.destroy()
    return directory


def get_model_and_processor(cache_dir=None):
    global _model, _processor
    if _model is None or _processor is None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            _processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
            _model = AutoModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir).to(device)
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return None, None
    return _processor, _model


def load_images_recursively(folder_path):
    if not os.path.exists(folder_path):
        return []
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_image_embeddings(image_paths, cache_dir=None):
    processor, model = get_model_and_processor(cache_dir)
    if processor is None or model is None:
        return []

    device = next(model.parameters()).device
    embeddings = []
    batch_size = st.session_state.batch_size

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
            except Exception as e:
                st.error(f"Error loading image {img_path}: {str(e)}")

        if not batch_images:
            continue

        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                batch_embeddings = image_features.cpu().numpy()
            embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"Error processing batch: {str(e)}")
            continue

    return embeddings


def process_images_in_chunks(image_paths, collection, cache_dir=None, chunk_size=1000):
    total_images = len(image_paths)
    processed_count = 0
    existing_ids = set(collection.get()["ids"])
    new_image_paths = [path for i, path in enumerate(image_paths) if str(i) not in existing_ids]

    if not new_image_paths:
        st.info("No new images to process")
        return

    progress_container = st.empty()
    status_container = st.empty()
    progress_bar = progress_container.progress(0)
    status_container.text(f"Processing {len(new_image_paths)} new images...")

    for i in range(0, len(new_image_paths), chunk_size):
        chunk_paths = new_image_paths[i:i + chunk_size]
        status_container.text(f"Processing chunk {i//chunk_size + 1}/{(len(new_image_paths) + chunk_size - 1)//chunk_size} ({processed_count}/{len(new_image_paths)} images)")

        embeddings = get_image_embeddings(chunk_paths, cache_dir=cache_dir)
        if not embeddings:
            continue

        try:
            start_idx = len(existing_ids) + i
            collection.add(
                embeddings=embeddings,
                metadatas=[{"path": str(path)} for path in chunk_paths],
                ids=[str(j + start_idx) for j in range(len(chunk_paths))]
            )
            processed_count += len(chunk_paths)
            progress_bar.progress(processed_count / len(new_image_paths))
        except Exception as e:
            st.error(f"Error storing chunk {i//chunk_size + 1}: {str(e)}")

    progress_container.empty()
    status_container.empty()
    st.success(f"Successfully processed {processed_count} images!")


def display_search_results(results: Dict[str, Any], max_images: int = 10):
    if not results or not results["metadatas"][0]:
        st.error("No results found")
        return

    st.session_state.search_results = results
    similar_images = [
        {"path": metadata["path"], "distance": distance}
        for metadata, distance in zip(results["metadatas"][0], results["distances"][0])
    ]
    similar_images.sort(key=lambda x: x["distance"])

    st.subheader(f"Top {min(max_images, len(similar_images))} Similar Images:")
    num_cols = 5
    num_rows = (len(similar_images[:max_images]) + num_cols - 1) // num_cols

    for row in range(num_rows):
        cols = st.columns(num_cols)
        for col_idx in range(num_cols):
            img_idx = row * num_cols + col_idx
            if img_idx < len(similar_images[:max_images]):
                with cols[col_idx]:
                    try:
                        st.image(similar_images[img_idx]["path"], caption=f"Match {img_idx+1} (Distance: {similar_images[img_idx]['distance']:.4f})")
                    except Exception as e:
                        st.error(f"Failed to display image {similar_images[img_idx]['path']}: {str(e)}")


def main():
    st.title("🔍 Image Search App")
    if 'model_cache_dir' not in st.session_state:
        st.session_state.model_cache_dir = os.environ.get('TRANSFORMERS_CACHE', '')
    if 'image_folder_path' not in st.session_state:
        st.session_state.image_folder_path = ''
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ''

    st.sidebar.header("⚙️ Settings")
    st.sidebar.subheader("🔎 Search Settings")
    st.session_state.num_results = st.sidebar.slider("Number of top images to retrieve", 1, 20, value=st.session_state.num_results)
    st.session_state.batch_size = st.sidebar.slider("Batch size for processing", 1, 64, value=st.session_state.batch_size)

    st.sidebar.subheader("💾 Model Cache Directory")
    model_cache_dir = st.sidebar.text_input("Path to cache directory", value=st.session_state.model_cache_dir)
    if st.sidebar.button("Browse Cache"):
        selected_dir = select_directory()
        if selected_dir:
            st.session_state.model_cache_dir = selected_dir
            st.rerun()

    st.sidebar.subheader("🖼️ Image Folder")
    folder_path = st.sidebar.text_input("Path to image folder", value=st.session_state.image_folder_path)
    if st.sidebar.button("Browse Images"):
        selected_dir = select_directory()
        if selected_dir:
            st.session_state.image_folder_path = selected_dir
            st.rerun()

    st.sidebar.subheader("📁 Embeddings Storage")
    embeddings_dir = st.sidebar.text_input("Path to embeddings directory", value=st.session_state.embeddings_dir)
    if st.sidebar.button("Browse Embeddings"):
        selected_dir = select_directory()
        if selected_dir:
            st.session_state.embeddings_dir = selected_dir
            st.rerun()

    try:
        client = PersistentClient(
            path=st.session_state.embeddings_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection(
            name="image_embeddings",
            metadata={"description": "Image embeddings from CLIP model"},
            embedding_function=None
        )
        total_images = len(collection.get()["ids"])
        if total_images > 0:
            st.sidebar.info(f"Found {total_images} existing embeddings")
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return

    if folder_path and st.sidebar.button("🚀 Process Images"):
        with st.spinner("Processing images..."):
            image_paths = load_images_recursively(folder_path)
            if not image_paths:
                st.error("No images found in the specified directory")
                return
            process_images_in_chunks(image_paths, collection, cache_dir=st.session_state.model_cache_dir)
            st.success("All embeddings saved to disk!")

    st.header("🔍 Search Images")
    query_type = st.radio("Select query type:", ["Text", "Image"])

    if query_type == "Text":
        query_text = st.text_input("Enter your text query:")
        if query_text and st.button("Search"):
            processor, model = get_model_and_processor(cache_dir=st.session_state.model_cache_dir)
            if processor is None or model is None:
                return
            device = next(model.parameters()).device
            inputs = processor(text=[query_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                query_embedding = text_features.cpu().squeeze().numpy()
            results = collection.query(query_embeddings=[query_embedding], n_results=st.session_state.num_results)
            display_search_results(results, max_images=st.session_state.num_results)

    else:
        query_image = st.file_uploader("Upload a query image", type=['png', 'jpg', 'jpeg'])
        if query_image and st.button("Search"):
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(query_image.getbuffer())
                temp_path = temp_file.name
            try:
                query_embedding = get_image_embeddings([temp_path], cache_dir=st.session_state.model_cache_dir)[0]
                results = collection.query(query_embeddings=[query_embedding], n_results=st.session_state.num_results)
                display_search_results(results, max_images=st.session_state.num_results)
            finally:
                os.remove(temp_path)


if __name__ == "__main__":
    main()
