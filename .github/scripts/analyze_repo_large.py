import os
import sys
import time
import math
from pathlib import Path
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "z-ai/glm-5.2"

# File extensions to include
INCLUDE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".cs", ".rb", ".php", ".swift", ".kt",
    ".yml", ".yaml", ".json", ".toml", ".md", ".sh", ".sql"
}
# Directories to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".next", "vendor", ".mypy_cache", ".pytest_cache",
    "coverage", "tests", "test", "spec", "docs"  # Skip tests/docs for core logic analysis
}
MAX_FILE_SIZE_KB = 100
MAX_CHUNK_TOKENS = 30000  # Safe limit per chunk to avoid context overflow
MAX_CHUNKS = 15  # Prevent infinite loops on massive monorepos


# ── File Collection ────────────────────────────────────────────────────────────
def collect_repo_files(root_path: str) -> list[dict]:
    files = []
    root = Path(root_path)

    for file_path in sorted(root.rglob("*")):
        if file_path.is_dir() or any(skip in file_path.parts for skip in SKIP_DIRS):
            continue

        suffix = file_path.suffix.lower()
        if suffix not in INCLUDE_EXTENSIONS and file_path.name not in INCLUDE_EXTENSIONS:
            continue

        size_kb = file_path.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            files.append({
                "path": str(file_path.relative_to(root)),
                "content": content,
                "tokens": len(content) // 4  # Rough estimate
            })
        except Exception:
            continue

    return files


# ── Chunking Logic ─────────────────────────────────────────────────────────────
def chunk_files(files: list[dict]) -> list[list[dict]]:
    chunks = []
    current_chunk = []
    current_tokens = 0

    for f in files:
        # If a single file is too big, skip it
        if f["tokens"] > MAX_CHUNK_TOKENS:
            continue

        if current_tokens + f["tokens"] > MAX_CHUNK_TOKENS:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

            # Safety cap on total chunks
            if len(chunks) >= MAX_CHUNKS:
                break

        current_chunk.append(f)
        current_tokens += f["tokens"]

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ── API Call with Retry ────────────────────────────────────────────────────────
def call_nim(client, messages, max_tokens=4096):
    """Calls the API with basic retry logic for rate limits."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,
                top_p=0.8,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < 2:
                print(f"  ⏳ Rate limited. Waiting 5 seconds...")
                time.sleep(5)
            else:
                raise e


# ── Main Execution ─────────────────────────────────────────────────────────────
def main():
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."

    print(f"\n{'=' * 60}")
    print(f"  LARGE REPO ANALYZER - {MODEL} on NVIDIA NIM")
    print(f"{'=' * 60}\n")

    # 1. Collect Files
    print("📁 Scanning repository...")
    files = collect_repo_files(repo_root)
    print(f"✅ Found {len(files)} analyzable files.\n")

    if not files:
        print("❌ No files found.")
        sys.exit(1)

    # 2. Chunk Files
    chunks = chunk_files(files)
    print(f"📦 Split into {len(chunks)} chunks for processing.\n")

    client = OpenAI(base_url=BASE_URL, api_key=NVIDIA_API_KEY)
    chunk_summaries = []

    # 3. Analyze Each Chunk
    for idx, chunk in enumerate(chunks):
        file_names = [f["path"] for f in chunk]
        print(f"🤖 Analyzing chunk {idx + 1}/{len(chunks)} ({len(chunk)} files)...")

        # Build chunk prompt
        contents = "\n\n".join(f"### {f['path']}\n```{Path(f['path']).suffix[1:]}\n{f['content']}\n```" for f in chunk)

        chunk_prompt = f"""You are analyzing a subset of a codebase. 
Files in this chunk: {', '.join(file_names)}

{contents}

Provide a concise analysis (max 500 words) covering:
1. **Purpose**: What do these files do?
2. **Bugs/Security**: Any logical errors, crashes, or security flaws?
3. **Code Smells**: Poor patterns, duplication, or complexity?
Be direct. Use bullet points."""

        summary = call_nim(client, [
            {"role": "system", "content": "You are an expert code reviewer. Be concise and precise."},
            {"role": "user", "content": chunk_prompt}
        ], max_tokens=1024)

        chunk_summaries.append(f"### Chunk {idx + 1} ({', '.join(file_names[:3])}...)\n{summary}")
        print(f"   ✅ Chunk {idx + 1} complete.\n")

        # Prevent rate limiting between chunks
        if idx < len(chunks) - 1:
            time.sleep(2)

            # 4. Final Synthesis
    print("🔗 Synthesizing final report from all chunks...\n")

    combined_summaries = "\n\n".join(chunk_summaries)

    # Truncate if synthesis prompt is too large
    if len(combined_summaries) > 100000:
        combined_summaries = combined_summaries[:100000] + "\n\n[Truncated due to size...]"

    synthesis_prompt = f"""You have analyzed a large repository in {len(chunks)} chunks. 
Here are the findings from each chunk:

{combined_summaries}

Now, synthesize these findings into a comprehensive, unified final report. 
Do not just list the chunks. Combine duplicate findings and identify cross-cutting architectural issues.

Structure your final report exactly like this:

# 🏗️ Repository Analysis Report

## 1. Executive Summary
(2-3 sentences on what the project is and its overall health)

## 2. Architecture & Design
(High-level patterns, how components interact, structural strengths/weaknesses)

## 3. 🚨 Critical Issues (Bugs & Security)
(List the most severe bugs, crashes, or security vulnerabilities found across all chunks. Include file paths.)

## 4. 🧹 Code Quality & Smells
(Major patterns of poor code, duplication, or technical debt)

## 5. ✅ What's Done Well
(Highlight good practices observed)

## 6. 🚀 Top 5 Actionable Recommendations
(Prioritized list of the most impactful changes to make immediately)

Format in clean Markdown."""

    final_report = call_nim(client, [
        {"role": "system",
         "content": "You are a Principal Software Architect. Write a professional, actionable report."},
        {"role": "user", "content": synthesis_prompt}
    ], max_tokens=4096)

    # 5. Output Results
    print("=" * 60)
    print("  FINAL REPORT GENERATED")
    print("=" * 60)

    output_path = Path("repo_analysis.md")
    output_path.write_text(final_report, encoding="utf-8")
    print(f"\n💾 Saved to: {output_path.resolve()}\n")

    # Set GitHub Actions outputs
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.getenv("GITHUB_OUTPUT"), "a") as f:
            f.write(f"analysis_complete=true\n")
            f.write(f"chunks_processed={len(chunks)}\n")


if __name__ == "__main__":
    main()