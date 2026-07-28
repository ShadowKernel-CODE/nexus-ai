import os
import json
import re
from typing import List, Tuple


def extract_text_from_file(file_path: str, file_type: str) -> Tuple[str, List[str]]:
    text = ""
    try:
        if file_type == ".txt":
            text = _extract_txt(file_path)
        elif file_type == ".pdf":
            text = _extract_pdf(file_path)
        elif file_type == ".docx":
            text = _extract_docx(file_path)
        elif file_type in (".png", ".jpg", ".jpeg", ".gif"):
            text = _extract_image_metadata(file_path, file_type)
        elif file_type in (".mp3", ".wav", ".m4a", ".webm", ".mp4"):
            text = _extract_audio_metadata(file_path, file_type)
        else:
            text = f"[File: {os.path.basename(file_path)} - file type {file_type} is stored but text extraction is not supported]"
    except Exception as e:
        text = f"[Error extracting text from {os.path.basename(file_path)}: {str(e)}]"

    chunks = chunk_text(text) if text else []
    return text, chunks


def _extract_txt(file_path: str) -> str:
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ""


def _extract_pdf(file_path: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF extraction error: {e}]"


def _extract_docx(file_path: str) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


def _extract_image_metadata(file_path: str, file_type: str) -> str:
    size = os.path.getsize(file_path)
    return f"[Image file: {os.path.basename(file_path)}, type: {file_type}, size: {size} bytes. Upload text descriptions or notes about this image as separate text files for better memory recall.]"


def _extract_audio_metadata(file_path: str, file_type: str) -> str:
    size = os.path.getsize(file_path)
    return f"[Audio file: {os.path.basename(file_path)}, type: {file_type}, size: {size} bytes. You can describe the content of this audio in chat to add it to the memory context.]"


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk)
                if overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-overlap // 5:] if len(words) > overlap // 5 else []
                    current_chunk = " ".join(overlap_words) + "\n\n" + para if overlap_words else para
                else:
                    current_chunk = para
            else:
                if len(para) > chunk_size:
                    words = para.split()
                    temp = ""
                    for word in words:
                        if len(temp) + len(word) + 1 <= chunk_size:
                            temp = f"{temp} {word}" if temp else word
                        else:
                            chunks.append(temp)
                            temp = word
                    current_chunk = temp
                else:
                    current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:chunk_size]]
