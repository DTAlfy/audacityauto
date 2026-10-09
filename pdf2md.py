#!/usr/bin/env python3
"""pdf2md: convierte PDFs en Markdown limpio usando modelos de IA locales.

Flujo:
  1. Extrae el PDF a Markdown "en bruto" con pymupdf4llm (sin IA).
  2. Lo trocea por páginas y pide a un modelo local que lo limpie y ordene.

Backends soportados (se detectan solos):
  - Ollama            http://localhost:11434
  - LM Studio         http://localhost:1234   (API compatible con OpenAI)
  - Cualquier servidor compatible con OpenAI (llama.cpp, vLLM, Jan...) con --url

Uso:
  python pdf2md.py --list-models
  python pdf2md.py documento.pdf
  python pdf2md.py *.pdf --model ollama:qwen2.5:7b -o salida/
  python pdf2md.py documento.pdf --no-ai
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

KNOWN_SERVERS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234",
}

SYSTEM_PROMPT = """Eres un conversor de documentos. Recibes Markdown extraído \
automáticamente de un PDF, con ruido: saltos de línea rotos, palabras cortadas \
con guion, cabeceras y pies de página repetidos, números de página sueltos y \
jerarquía de títulos incorrecta.

Devuelve ÚNICAMENTE el Markdown limpio:
- Une párrafos y palabras partidas por saltos de línea.
- Elimina cabeceras/pies de página repetidos y números de página sueltos.
- Usa una jerarquía de títulos coherente (#, ##, ###).
- Conserva listas, tablas (en formato Markdown), enlaces y bloques de código.
- NO resumas, NO traduzcas, NO añadas contenido ni comentarios propios.
- No envuelvas la respuesta en ```markdown."""


# ---------------------------------------------------------------- HTTP helpers

def _http(url, payload=None, timeout=5):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ------------------------------------------------------------ model discovery

def list_models(extra_url=None):
    """Devuelve [(backend, base_url, modelo)] de los servidores que respondan."""
    found = []

    try:
        tags = _http(f"{KNOWN_SERVERS['ollama']}/api/tags")
        for m in tags.get("models", []):
            found.append(("ollama", KNOWN_SERVERS["ollama"], m["name"]))
    except (urllib.error.URLError, OSError, ValueError):
        pass

    openai_urls = [("lmstudio", KNOWN_SERVERS["lmstudio"])]
    if extra_url:
        openai_urls.append(("openai", extra_url.rstrip("/")))
    for backend, base in openai_urls:
        try:
            data = _http(f"{base}/v1/models")
            for m in data.get("data", []):
                found.append((backend, base, m["id"]))
        except (urllib.error.URLError, OSError, ValueError):
            pass

    return found


def pick_model(spec, available):
    """spec = 'backend:modelo', 'modelo' o None (el primero disponible)."""
    if not available:
        sys.exit(
            "No se encontró ningún modelo local.\n"
            "  - Ollama:    instala desde https://ollama.com y ejecuta `ollama pull qwen2.5:7b`\n"
            "  - LM Studio: carga un modelo y activa 'Start Server' (puerto 1234)\n"
            "  - O usa --no-ai para sólo extraer el texto."
        )
    if spec is None:
        return available[0]
    backend, _, name = spec.partition(":") if spec.split(":")[0] in (
        "ollama", "lmstudio", "openai") else (None, None, spec)
    for b, url, m in available:
        if m == name and (backend is None or b == backend):
            return b, url, m
    sys.exit(f"Modelo '{spec}' no encontrado. Usa --list-models para ver los disponibles.")


# ------------------------------------------------------------------ LLM calls

def clean_with_llm(text, backend, base_url, model, timeout):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if backend == "ollama":
        resp = _http(
            f"{base_url}/api/chat",
            {"model": model, "messages": messages, "stream": False,
             "options": {"temperature": 0.1, "num_ctx": 16384}},
            timeout=timeout,
        )
        out = resp["message"]["content"]
    else:
        resp = _http(
            f"{base_url}/v1/chat/completions",
            {"model": model, "messages": messages, "temperature": 0.1},
            timeout=timeout,
        )
        out = resp["choices"][0]["message"]["content"]
    return strip_fences(out)


def strip_fences(text):
    text = text.strip()
    # Algunos modelos de razonamiento devuelven <think>...</think> antes.
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    return text.strip()


# ----------------------------------------------------------------- extraction

def extract_pages(pdf_path):
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    return [p["text"] for p in pages]


def chunk_pages(pages, max_chars):
    """Agrupa páginas en bloques de como máximo max_chars caracteres."""
    chunks, current = [], ""
    for page in pages:
        if current and len(current) + len(page) > max_chars:
            chunks.append(current)
            current = ""
        current += page + "\n\n"
    if current.strip():
        chunks.append(current)
    return chunks


# ----------------------------------------------------------------------- main

def convert(pdf_path, out_dir, model_info, args):
    pages = extract_pages(pdf_path)
    if model_info is None:
        result = "\n\n".join(pages)
    else:
        chunks = chunk_pages(pages, args.chunk_chars)
        cleaned = []
        for i, chunk in enumerate(chunks, 1):
            print(f"  bloque {i}/{len(chunks)}...", flush=True)
            cleaned.append(clean_with_llm(chunk, *model_info, timeout=args.timeout))
        result = "\n\n".join(cleaned)

    out_path = out_dir / (pdf_path.stem + ".md")
    out_path.write_text(result.strip() + "\n", encoding="utf-8")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Convierte PDFs en Markdown limpio con IA local.")
    p.add_argument("pdfs", nargs="*", type=Path, help="Archivos PDF a convertir")
    p.add_argument("-o", "--out", type=Path, help="Carpeta de salida (por defecto, junto al PDF)")
    p.add_argument("-m", "--model", help="Modelo a usar, p.ej. 'ollama:llama3.1:8b' o 'qwen2.5:7b'")
    p.add_argument("--url", help="URL de otro servidor compatible con OpenAI, p.ej. http://localhost:8080")
    p.add_argument("--list-models", action="store_true", help="Muestra los modelos locales detectados")
    p.add_argument("--no-ai", action="store_true", help="Sólo extracción, sin limpieza con IA")
    p.add_argument("--chunk-chars", type=int, default=6000, help="Tamaño de bloque enviado al modelo")
    p.add_argument("--timeout", type=int, default=600, help="Segundos máximos por bloque")
    args = p.parse_args()

    if args.list_models:
        models = list_models(args.url)
        if not models:
            print("No se detectó ningún servidor de modelos (Ollama / LM Studio).")
        for backend, url, name in models:
            print(f"{backend}:{name:40s} ({url})")
        return

    if not args.pdfs:
        p.error("indica al menos un PDF (o usa --list-models)")

    model_info = None
    if not args.no_ai:
        model_info = pick_model(args.model, list_models(args.url))
        print(f"Usando {model_info[0]}:{model_info[2]}")

    for pdf in args.pdfs:
        if not pdf.is_file():
            print(f"[omitido] {pdf} no existe")
            continue
        out_dir = args.out or pdf.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Convirtiendo {pdf}...")
        print(f"  -> {convert(pdf, out_dir, model_info, args)}")


if __name__ == "__main__":
    main()
