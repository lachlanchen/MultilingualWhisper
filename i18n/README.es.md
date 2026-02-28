[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Un generador de subtítulos listo para usar, construido sobre OpenAI Whisper y ampliado con detección y refinamiento precisos del idioma por segmento para videos con idiomas mixtos.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Resumen

`MultilingualWhisper` es una canalización CLI en Python centrada en [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Combina:

- Silero VAD para segmentación de voz
- OpenAI Whisper para transcripción y predicción inicial de idioma
- Lingua para refinamiento del idioma basado en texto
- FFmpeg para extracción, normalización y manejo de medios

Los resultados principales son archivos de subtítulos en `.srt` y `.json`, además de audio `.wav` extraído y normalizado.

### De un Vistazo

| Elemento | Detalles |
|---|---|
| Punto de entrada principal | `vad_lang_subtitle.py` |
| Entrada | Video/audio compatible con FFmpeg |
| Salida | `*.wav`, `*.srt`, `*.json` |
| Flujo central | VAD -> Whisper -> Lingua -> refinamiento |
| Caso de uso típico | Generación de subtítulos multilingües |

---

## 🚀 Características Principales

- **Canalización Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) divide el audio en segmentos de voz y luego Whisper transcribe cada bloque.

- **Detección de idioma de grano fino**  
  Usa [Lingua](https://github.com/pemistahl/lingua-java) junto con el detector propio de Whisper para etiquetar cada segmento (incluso palabras individuales) con códigos de idioma ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Refinamiento inteligente de segmentos**  
  La limpieza de marcas de tiempo asegura que no haya huecos ni solapamientos. La división por puntuación separa transcripciones largas en comas, puntos, signos de interrogación, etc. Las fusiones de VAD realinean palabras de vuelta a bloques VAD para subtítulos más fluidos. La segmentación sensible a la longitud aplica límites específicos por idioma.

- **Subtítulos multilingües**  
  Genera tanto `.srt` como `.json`, preservando etiquetas de idioma por segmento para que puedas aplicar estilo o filtrar por idioma en reproductores o editores posteriores.

- **Manejo robusto de medios**  
  Extrae y normaliza audio automáticamente mediante FFmpeg, intenta reparar contenedores dañados y aplica normalización dinámica (`dynaudnorm`) para transcripciones más claras.

---

## 🗂 Estructura del Proyecto

```text
.
├── README.md
├── vad_lang_subtitle.py               # Main pipeline: VAD -> Whisper -> Lingua -> refine -> save
├── vad_lang_subtitle.py.old           # Legacy prototype
├── vad_lang_subtitle.py.20250706      # Historical snapshot
├── vad_lang_subtitle.py.shorterlength # Alternative historical variant
├── vad_lang_subtitle.py.shorterlength2# Alternative historical variant
├── vad_lang_subtitle.srt              # Example output
├── vad_lang_subtitle.json             # Example JSON
├── .github/
│   └── FUNDING.yml                    # Sponsor links
├── archived/                          # Old experiments/prototypes
├── data/                              # Optional sample media + generated outputs
├── figs/                              # Branding assets (banner/logo)
├── i18n/                              # Translation/readme workspace (currently present, empty)
└── .auto-readme-work/                 # README generation workspace artifacts
```

> ⚠️ Nota: El README anterior hacía referencia a `requirements.txt`, pero actualmente falta en la raíz del repositorio.

---

## ✅ Requisitos Previos

- Python `3.10+` (probado con entornos 3.x modernos)
- `ffmpeg` instalado y disponible en `PATH`
- CPU/GPU + RAM suficientes para el modelo Whisper seleccionado (para `large`, se recomienda GPU)
- Acceso a Internet en la primera ejecución para descargar pesos del modelo Whisper y recursos de Silero VAD (`torch.hub`)

Los paquetes de Python usados por el script incluyen:

- `torch`
- `torchaudio`
- `whisper` (paquete Python de OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

---

## 🔧 Instalación

1. **Clona este repositorio**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **Crea y activa un entorno virtual**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instala las dependencias**

```bash
pip install -r requirements.txt
```

Si `requirements.txt` sigue ausente en tu copia, instala manualmente las dependencias principales de ejecución:

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
| `--force` | — | No | Reejecutar incluso si `.wav`, `.srt` o `.json` ya existen |

### Comportamiento del Procesamiento

- Los nombres de salida se derivan de la ruta base de entrada.
- Para `input.mp4`, las salidas son `input.wav` (audio normalizado), `input.srt` (subtítulos con marcas de tiempo) y `input.json` (metadatos incluyendo `start`, `end`, `lang`, `text` y, opcionalmente, tiempos por palabra).
- Si ya existen `.srt` o `.json`, se omite el proceso salvo que se establezca `--force`.

---

## ⚙️ Configuración

La configuración actual se basa principalmente en la CLI y en valores predeterminados del código:

- Modelo Whisper: `--whisper-model` (predeterminado `large`)
- Frecuencia de muestreo: fijada en `16000` para el procesamiento
- Extracción con FFmpeg: WAV mono, `44100 Hz`, con `dynaudnorm=f=100`
- Detector Lingua: inicializado para `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` en el flujo principal
- Los códigos de idioma permitidos para el filtrado del lado Whisper incluyen `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` en los valores auxiliares por defecto

Nota de suposición: las listas de idioma en los valores auxiliares por defecto y la configuración principal del detector no son totalmente idénticas; este README conserva el comportamiento actual tal como está implementado.

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

Ejecutar sobre una entrada solo-audio compatible con FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 Notas de Desarrollo

- El script activo canónico es `vad_lang_subtitle.py`.
- Los archivos históricos (`*.old`, `*.shorterlength*`, `archived/`) son útiles como referencia, pero parecen no canónicos.
- Actualmente no hay andamiaje de proyecto empaquetado (`pyproject.toml`, `setup.py`) ni una suite de CI/pruebas confirmada en el repositorio.
- `data/` contiene artefactos de medios de muestra de gran tamaño; tenlo en cuenta por el tamaño del repositorio y el uso de disco local durante pruebas.
- `clean_subtitles_dict()` existe en el código, pero actualmente no se invoca en la canalización principal.

---

## 🩺 Solución de Problemas

| Síntoma | Qué hacer |
|---|---|
| `ffmpeg: command not found` | Instala FFmpeg y verifica con `ffmpeg -version`. |
| La primera ejecución es muy lenta o parece bloqueada | Las descargas iniciales de modelos (Whisper + Silero) pueden tardar; las siguientes ejecuciones son más rápidas. |
| Errores de CUDA / GPU | Prueba el fallback a CPU usando un modelo Whisper más pequeño (`small`, `base`, `tiny`) y asegúrate de tener una compilación de PyTorch compatible con tu entorno. |
| Los archivos de salida no se regeneran | Usa `--force` para sobrescribir archivos derivados existentes. |
| `pip install -r requirements.txt` falla porque no se encuentra el archivo | Usa el comando de instalación manual de dependencias mostrado en Instalación. |
| Etiquetado de idioma inexacto en segmentos cortos | Puede ocurrir en tramos extremadamente cortos/ruidosos; la lógica actual combina Whisper y Lingua, pero todavía hay casos límite. |

---

## 🗺 Hoja de Ruta

- Añadir y mantener un `requirements.txt` o `pyproject.toml` fijado.
- Añadir pruebas automatizadas para la segmentación y la lógica de limpieza de marcas de tiempo.
- Añadir documentación de benchmark y evaluación de calidad para casos límite multilingües.
- Añadir soporte opcional de archivo de configuración en lugar de comportamiento solo con valores por defecto en código.
- Ampliar el conjunto de README i18n en `i18n/` y mantener sincronizadas las barras de idioma.

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
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) por detección robusta de actividad de voz
- [Lingua](https://github.com/pemistahl/lingua-java) por identificación de idioma de alta precisión

---

## 🤝 Contribuir

1. Haz un fork y clona
2. Crea una rama: `git checkout -b feat/your-idea`
3. Haz commit y push
4. Abre un PR

---

## 📄 Licencia

MIT © Lachlan Chen
