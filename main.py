import os
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from markitdown import MarkItDown

app = FastAPI(title="Markdown Converter API")

# TODO: after you deploy the frontend, replace "*" with your Vercel URL
# e.g. ["https://your-app.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

md = MarkItDown()

ALLOWED_EXT = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".txt", ".csv", ".json"}


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/convert")
async def convert_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        return {"filename": file.filename, "markdown": result.text_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)
