"""
deploy_to_spaces.py
====================
Deploys this project to HuggingFace Spaces.

Usage:
    python deploy_to_spaces.py --token YOUR_HF_TOKEN

Get your token at: https://huggingface.co/settings/tokens
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def deploy(hf_token: str, username: str = "dipanshuverma98", space_name: str = "sarcasm-detection"):
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("❌ huggingface_hub not installed.")
        print("   Run: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=hf_token)

    # Verify token
    try:
        user = api.whoami()
        print(f"✅ Logged in as: {user['name']}")
    except Exception as e:
        print(f"❌ Invalid token: {e}")
        sys.exit(1)

    repo_id = f"{username}/{space_name}"
    print(f"\n📦 Creating Space: {repo_id}")

    # Create the Space
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            token=hf_token,
            exist_ok=True,
            private=False,
        )
        print(f"✅ Space created: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"❌ Could not create Space: {e}")
        sys.exit(1)

    # Prepare upload folder — Spaces needs README.md with YAML frontmatter
    project_root = Path(__file__).parent
    spaces_readme = project_root / "SPACES_README.md"
    main_readme   = project_root / "README.md"

    # Temporarily rename SPACES_README → README for upload
    shutil.copy(str(spaces_readme), str(main_readme))

    print("\n📤 Uploading files to Space...")
    try:
        upload_folder(
            folder_path=str(project_root),
            repo_id=repo_id,
            repo_type="space",
            token=hf_token,
            ignore_patterns=[
                "*.pt", "*.bin", "checkpoints/**", "eval_output/**",
                "data/*.csv", "data/*.zip", ".git/**", "__pycache__/**",
                "*.ipynb_checkpoints/**", ".venv/**", "venv/**",
                "SPACES_README.md", "deploy_to_spaces.py",
            ],
        )
        print(f"\n🎉 Deployed successfully!")
        print(f"🔗 Live at: https://huggingface.co/spaces/{repo_id}")
        print(f"⏳ First build takes ~2-3 minutes. Refresh the page.")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",    required=True, help="HuggingFace API token (write access)")
    parser.add_argument("--username", default="dipanshuverma98", help="HuggingFace username")
    parser.add_argument("--space",    default="sarcasm-detection", help="Space name")
    args = parser.parse_args()

    deploy(hf_token=args.token, username=args.username, space_name=args.space)
