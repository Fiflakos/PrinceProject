# agents/interface_agent.py

import streamlit as st
from retrieval_agent import LocalRetrievalAgent
import requests
import os

import os
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

#HF_MODEL_ENDPOINT = "https://api-inference.huggingface.co/models/google/flan-t5-base"

BASE_DIR = "data_cleaned"
retriever = LocalRetrievalAgent(corpus_dir=BASE_DIR)

st.title("🪖 WW1 Historical Assistant")
st.markdown("Ask questions about letters and diaries from WW1.")

# Step 1: Get list of available documents
def get_all_txt_files(base_dir):
    files = []
    for subdir in ["Letters", "Diaries", "Others"]:
        folder_path = os.path.join(base_dir, subdir)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    files.append((subdir, filename))
    return files

available_files = get_all_txt_files(BASE_DIR)
file_options = [f"{folder}/{name}" for folder, name in available_files]

# Step 2: User selects a file from dropdown
selected_file = st.selectbox("📂 Select a document", file_options)

# Step 3: Read file content automatically
if selected_file:
    folder, filename = selected_file.split("/", 1)
    file_path = os.path.join(BASE_DIR, folder, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    st.session_state["documents"] = [{"text": content, "meta": {"file": filename}}]
    st.success(f"Loaded: {filename}")

# Step 4: Question input and API call
question = st.text_input("💬 Ask a question:")
if question and st.session_state.get("documents"):
    context = ""
    for doc in st.session_state["documents"][:3]:
        file = doc["meta"].get("file", "Unknown")
        chunk = doc["text"].replace("\n", " ").strip()
        context += f"{file}:\n{chunk}\n\n"

    prompt = f"Answer the question based on the following WW1 letters or diaries.\n\nContext:\n{context}\n\nQuestion: {question}"

    with st.spinner("🧠 Thinking..."):
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 256}
        }

        response = requests.post(HF_MODEL_ENDPOINT, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                answer = result[0]["generated_text"]
            elif isinstance(result, dict) and "generated_text" in result:
                answer = result["generated_text"]
            else:
                answer = "No answer generated."
            st.markdown(f"**📜 Answer:** {answer}")
        else:
            st.error(f"API Error: {response.status_code} — {response.text}")
else:
    st.info("Select a document and ask a question.")
