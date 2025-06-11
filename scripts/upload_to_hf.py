"""
Script to upload your fine-tuned model to Hugging Face Hub.
Make sure you are logged in (`huggingface-cli login`) first.
"""

from huggingface_hub import HfApi, HfFolder, Repository
import os

def upload_model(model_dir: str, repo_name: str):
    os.system("huggingface-cli login")
    api = HfApi()
    username = api.whoami()["name"]
    full_repo = f"{username}/{repo_name}"

    print(f"📦 Uploading to Hugging Face Hub: {full_repo}")
    api.create_repo(name=repo_name, exist_ok=True)
    repo = Repository(local_dir=model_dir, clone_from=full_repo)
    repo.push_to_hub(commit_message="Upload fine-tuned HistBERT model")

if __name__ == "__main__":
    upload_model("models/hf_model", "ww1-histbert")
