# pdf2md

Convierte PDFs en Markdown limpio y ordenado usando modelos de IA **locales** (en tu PC).

1. Extrae el texto del PDF con `pymupdf4llm` (rápido, sin IA).
2. Envía el texto por bloques a un modelo local que lo limpia: une párrafos rotos,
   quita cabeceras/pies y números de página, y ordena los títulos. No resume ni inventa.

## Instalación

```bash
pip install -r requirements.txt
```

Necesitas al menos un servidor de modelos local:

- **Ollama** (https://ollama.com): `ollama pull qwen2.5:7b`
- **LM Studio**: carga un modelo y pulsa *Start Server* (puerto 1234)
- Otro servidor compatible con OpenAI (llama.cpp, vLLM, Jan…): pásalo con `--url`

## Uso

```bash
python pdf2md.py --list-models                      # ver qué modelos tienes
python pdf2md.py informe.pdf                        # usa el primer modelo detectado
python pdf2md.py *.pdf -m ollama:qwen2.5:7b -o md/  # varios PDFs, modelo concreto
python pdf2md.py informe.pdf --no-ai                # sólo extracción, sin IA
```

Opciones útiles: `--chunk-chars` (tamaño de bloque; bájalo si tu modelo tiene poco contexto)
y `--timeout` (segundos por bloque).
