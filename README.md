# 🖥️ ScribeDoc Backend API

A lightweight, high-performance Python web API built with **FastAPI** that wraps around **Microsoft's MarkItDown** engine. It accepts document file uploads, converts them on the fly, and returns clean, LLM-optimized Markdown text.

---

## 💡 How It Works (The Easy Version)

Think of this backend server as a digital translator:
1. Your frontend website sends a raw document (like a PDF or Word file) over the network.
2. This API accepts the file, writes it to a temporary safe buffer on your disk, and hands it to Microsoft's layout engine.
3. The engine strips out all background formatting clutter (fonts, layouts, metadata) but keeps the important semantic structures (headers, lists, tables).
4. The server returns clean Markdown text back to your frontend interface and securely deletes the temporary file from the disk.

---

## 🛠️ Step-by-Step Installation Process

Follow these simple steps to run this server locally on your computer:

### 1. Initialize Your Environment
Open your terminal inside this folder and activate your Python virtual environment to keep your packages isolated:
```bash


python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload