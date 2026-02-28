[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Un generador de subtítulos listo para usar, construido sobre OpenAI Whisper y ampliado con detección y refinamiento precisos del idioma por segmento para videos con idiomas mezclados.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Tabla de Contenidos

- [Resumen](#-resumen)
- [Vista Rápida](#vista-rápida)
- [Funciones Principales](#-funciones-principales)
- [Flujo del Pipeline](#-flujo-del-pipeline)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Configuración](#-configuración)
- [Formato de Salida](#-formato-de-salida)
- [Ejemplos](#-ejemplos)
- [Notas de Desarrollo](#-notas-de-desarrollo)
- [Solución de Problemas](#-solución-de-problemas)
- [Limitaciones y Supuestos Conocidos](#-limitaciones-y-supuestos-conocidos)
- [Hoja de Ruta](#-hoja-de-ruta)
- [Soporte](#-soporte)
- [Agradecimientos](#-agradecimientos)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## ✨ Resumen

`MultilingualWhisper` es un pipeline CLI en Python centrado en [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Combina:

- Silero VAD para segmentación de voz
- OpenAI Whisper para transcripción y predicción inicial de idioma
- Lingua para refinamiento del idioma basado en texto
- FFmpeg para extracción, normalización y manejo multimedia

Las salidas principales son archivos de subtítulos en `.srt` y `.json`, además de audio `.wav` extraído y normalizado.

### Vista Rápida

| Elemento | Detalles |
|---|---|
| Punto de entrada principal | `vad_lang_subtitle.py` |
| Entrada | Video/audio compatible con FFmpeg |
| Salida | `*.wav`, `*.srt`, `*.json` |
| Flujo principal | VAD -> Whisper -> Lingua -> refinamiento |
| Caso de uso típico | Generación de subtítulos en idiomas mixtos |

---

## 🚀 Funciones Principales

- **Pipeline Silero VAD -> Whisper**
  La Detección de Actividad de Voz (VAD) divide el audio en segmentos de voz y luego Whisper transcribe cada fragmento.

- **Detección de idioma de grano fino**
  Usa [Lingua](https://github.com/pemistahl/lingua-java) junto con el detector propio de Whisper para etiquetar cada segmento (incluso palabras individuales) con códigos de idioma ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Refinamiento inteligente de segmentos**
  La limpieza de marcas de tiempo garantiza que no haya huecos ni solapamientos. La división por puntuación corta transcripciones largas en comas, puntos, signos de interrogación, etc. La fusión VAD realinea palabras con bloques VAD para subtítulos más fluidos. La segmentación sensible a longitud aplica límites específicos por idioma.

- **Subtítulos multilingües**
  Genera tanto `.srt` como `.json`, conservando etiquetas de idioma por segmento para poder aplicar estilos o filtrar por idioma en reproductores o editores posteriores.

- **Manejo multimedia robusto**
  Extrae y normaliza audio automáticamente mediante FFmpeg, intenta reparar contenedores dañados y aplica normalización dinámica (`dynaudnorm`) para transcripciones más claras.

---

## 🔁 Flujo del Pipeline

```text
Input media
  -> FFmpeg extract + normalize (.wav)
  -> Silero VAD speech timestamps
  -> Whisper transcription + language prediction
  -> Lingua segment language refinement
  -> Segment merge/split/timestamp cleanup
  -> Length-aware subtitle refinement
  -> Output .srt + .json
```

Ruta principal de ejecución en `vad_lang_subtitle.py`:

1. Analiza argumentos CLI (`--video-path`, `--whisper-model`, `--force`).
2. Resuelve rutas de salida desde el nombre base de entrada.
3. Extrae/normaliza audio mediante FFmpeg.
4. Carga Silero VAD (`torch.hub`) y el modelo Whisper.
5. Primera pasada de transcripción sobre fragmentos VAD.
6. Fusiona/refina segmentos y luego ejecuta una segunda pasada de transcripción sobre tramos fusionados.
7. Aplica reducción de longitud de subtítulos y limpieza de marcas de tiempo.
8. Guarda `.srt` y `.json`.

---

## 🗂 Estructura del Proyecto

```text
.
├── README.md
├── vad_lang_subtitle.py                # Main pipeline: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old            # Legacy prototype
├── vad_lang_subtitle.py.20250706       # Historical snapshot
├── vad_lang_subtitle.py.shorterlength  # Alternative historical variant
├── vad_lang_subtitle.py.shorterlength2 # Alternative historical variant
├── vad_lang_subtitle.srt               # Example output
├── vad_lang_subtitle.json              # Example JSON
├── .github/
│   └── FUNDING.yml                     # Sponsor links
├── archived/
│   ├── vad.py
│   ├── vad_lang.py
│   ├── vad_lang_subtitle.py
│   ├── decode_audio.py
│   ├── decode_audio_v2.py
│   ├── text_language_detect.py
│   └── trans_with_lang.py
├── data/                               # Optional sample media + generated outputs
├── figs/                               # Branding assets (banner/logo)
├── i18n/                               # Existing multilingual README files
└── .auto-readme-work/                  # README generation workspace artifacts
```

> ⚠️ Nota: El README anterior hacía referencia a `requirements.txt`, pero actualmente no existe en la raíz del repositorio.

---

## ✅ Requisitos Previos

- Python `3.10+` (probado con entornos 3.x modernos)
- `ffmpeg` instalado y disponible en `PATH`
- CPU/GPU + RAM suficientes para el modelo Whisper seleccionado (para `large`, se recomienda fuertemente GPU)
- Acceso a Internet en la primera ejecución para obtener los pesos del modelo Whisper y los recursos de Silero VAD (`torch.hub`)

Los paquetes de Python usados por el script incluyen:

- `torch`
- `torchaudio`
- `whisper` (paquete Python de OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

Comandos rápidos de verificación:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Instalación

1. **Clonar este repositorio**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **Crear y activar un entorno virtual**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

Si `requirements.txt` sigue ausente en tu copia local, instala manualmente las dependencias principales de ejecución:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Y asegúrate de que FFmpeg esté instalado a nivel de sistema.

---

## 🛠 Uso

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Opciones CLI

| Flag | Alias | Obligatorio | Descripción |
|---|---|---|---|
| `--video-path` | `-t` | Sí | Ruta del medio de entrada (video/audio compatible con FFmpeg) |
| `--whisper-model` | — | No | Nombre del modelo Whisper (predeterminado: `large`) |
| `--force` | — | No | Reejecuta incluso si ya existen `.wav`, `.srt` o `.json` |

### Comportamiento del Procesamiento

- Los nombres de salida se derivan de la ruta base de entrada.
- Para `input.mp4`, las salidas son `input.wav` (audio normalizado), `input.srt` (subtítulos con marcas de tiempo) e `input.json` (metadatos incluyendo `start`, `end`, `lang`, `text` y, opcionalmente, tiempos por palabra).
- Si existen `.srt` o `.json`, se omite el procesamiento a menos que se establezca `--force`.

---

## ⚙️ Configuración

La configuración actual se basa principalmente en CLI y valores predeterminados del código:

| Área de Configuración | Comportamiento Actual |
|---|---|
| Modelo Whisper | `--whisper-model` (predeterminado `large`) |
| Frecuencia de muestreo de procesamiento | Fijada en `16000` para procesamiento VAD/transcripción |
| Extracción FFmpeg | WAV mono, `44100 Hz`, con `dynaudnorm=f=100` |
| Detector Lingua | Inicializado para `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` en el flujo principal |
| Valores predeterminados del helper de filtrado en Whisper | Incluye `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Nota de supuesto: las listas de idiomas en los valores predeterminados del helper y la configuración principal del detector no son totalmente idénticas; este README conserva el comportamiento actual tal como está implementado.

---

## 📦 Formato de Salida

La herramienta escribe dos artefactos de subtítulos por cada medio de entrada:

- `*.srt`: Texto de subtítulos estándar con marcas de tiempo `HH:MM:SS,mmm`.
- `*.json`: Lista estructurada de subtítulos que contiene marcas de tiempo formateadas y etiquetas de idioma.

Forma típica de un segmento JSON:

```json
{
  "start": "00:00:01,234",
  "end": "00:00:03,456",
  "lang": "en",
  "text": "Hello world",
  "words": [
    {
      "word": " Hello",
      "start": 1234,
      "end": 1678,
      "probability": 0.98
    }
  ]
}
```

Notas:

- `start`/`end` se serializan como cadenas estilo SRT en la salida JSON.
- `words` puede estar presente según la etapa de procesamiento/refinamiento del segmento.
- Un valor `lang` de `und` puede aparecer en tramos con idioma incierto.

---

## 🧪 Ejemplos

Ejecutar sobre un MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Ejecutar sobre un MOV y forzar sobrescritura:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Ejecutar sobre una entrada solo de audio compatible con FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Ejemplo de procesamiento por lotes en shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Notas de Desarrollo

- El script activo canónico es `vad_lang_subtitle.py`.
- Los archivos históricos (`*.old`, `*.shorterlength*`, `archived/`) son útiles como referencia, pero parecen no canónicos.
- Actualmente no hay scaffolding de proyecto empaquetado (`pyproject.toml`, `setup.py`) ni suite de CI/tests confirmada en el repositorio.
- `data/` contiene artefactos multimedia de ejemplo grandes; ten en cuenta el tamaño del repositorio y el uso de disco local durante pruebas.
- `clean_subtitles_dict()` existe en el código, pero actualmente no se invoca desde el pipeline principal.
- `--force` es el mecanismo actual para garantizar la regeneración de salidas durante el ajuste iterativo.

Bucle de desarrollo local sugerido:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Usa un modelo más pequeño (`tiny`/`base`/`small`) mientras iteras y luego cambia a `large` para la calidad final de salida.

---

## 🩺 Solución de Problemas

| Síntoma | Qué hacer |
|---|---|
| `ffmpeg: command not found` | Instala FFmpeg y verifica con `ffmpeg -version`. |
| La primera ejecución es muy lenta o parece bloqueada | Las descargas iniciales de modelos (Whisper + Silero) pueden tardar; las siguientes ejecuciones son más rápidas. |
| Errores de CUDA / GPU | Prueba con CPU usando un modelo Whisper más pequeño (`small`, `base`, `tiny`) y asegúrate de tener una compilación de PyTorch compatible con tu entorno. |
| Los archivos de salida no se regeneran | Usa `--force` para sobrescribir archivos derivados existentes. |
| `pip install -r requirements.txt` falla porque no se encuentra el archivo | Usa el comando de instalación manual de dependencias mostrado en Instalación. |
| Etiquetado de idioma inexacto en segmentos cortos | Puede ocurrir en tramos extremadamente cortos o con ruido; la lógica actual combina Whisper y Lingua, pero aún hay casos límite. |
| Salida de subtítulos vacía o casi vacía | Confirma que la entrada tenga voz, inspecciona el `.wav` extraído y vuelve a intentar con `--force` tras validar la extracción con FFmpeg. |
| Cambios de idioma inesperados entre líneas vecinas | Puede ocurrir en segmentos muy cortos; considera post-fusión en herramientas posteriores por idioma y duración mínima. |

---

## ⚠️ Limitaciones y Supuestos Conocidos

- El manifiesto de dependencias no está confirmado (`requirements.txt`, `pyproject.toml` y `setup.py` están ausentes en la raíz del repositorio al momento de escribir esto).
- La licencia se declara en el README como MIT, pero actualmente no hay un archivo `LICENSE` independiente.
- Lingua se inicializa explícitamente con `EN/ZH/JA/AR` en el flujo principal, mientras que los valores predeterminados del helper incluyen más códigos candidatos.
- Actualmente no hay pruebas/benchmarks automatizados confirmados, por lo que la validación es principalmente manual.
- Hay scripts históricos en la raíz y en `archived/`; solo `vad_lang_subtitle.py` debe tratarse como activo salvo que se esté experimentando intencionalmente.

---

## 🗺 Hoja de Ruta

- Añadir y mantener un `requirements.txt` o `pyproject.toml` con versiones fijadas.
- Añadir pruebas automatizadas para la lógica de segmentación y limpieza de marcas de tiempo.
- Añadir documentación de benchmarks y evaluación de calidad para casos límite multilingües.
- Añadir soporte opcional de archivo de configuración en lugar de depender solo de valores predeterminados en código.
- Ampliar el conjunto de README i18n en `i18n/` y mantener sincronizadas las barras de idioma.
- Aclarar y unificar el comportamiento de selección de idioma entre la configuración del detector y los valores predeterminados del helper.
- Añadir un archivo `LICENSE` formal para que coincida con la declaración del README.

---

## 💖 Soporte

Si este proyecto te ayuda, puedes apoyar el desarrollo mediante:

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Sitio personal: https://lazying.art
- Chat/comunidad: https://chat.lazying.art
- Centro de ideas/proyectos: https://onlyideas.art

---

## 🔗 Agradecimientos

- [OpenAI Whisper](https://github.com/openai/whisper) por speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) por la detección robusta de actividad de voz
- [Lingua](https://github.com/pemistahl/lingua-java) por la identificación de idioma de alta precisión

---

## 🤝 Contribuir

1. Haz un fork y clona
2. Crea una rama: `git checkout -b feat/your-idea`
3. Haz commit y push
4. Abre un PR

Para cambios sustanciales, incluye:

- Una breve descripción del cambio esperado de comportamiento
- Un ejemplo de comando reproducible
- Fragmentos de subtítulos antes/después cuando corresponda

---

## 📄 Licencia

MIT © Lachlan Chen
