[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Un generador de subtítulos listo para usar basado en OpenAI Whisper, ampliado con detección y refinamiento precisos de idioma por segmento para vídeos con idiomas mezclados.

> Genera subtítulos multilingües más limpios desde contenido real con mezcla de idiomas, usando segmentación consciente del idioma.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Interface](https://img.shields.io/badge/Interface-CLI-1F6FEB)
![Output](https://img.shields.io/badge/Output-SRT%20%7C%20JSON-0A7F5A)
![Workflow](https://img.shields.io/badge/Flow-Silero%20%3E%20Whisper%20%3E%20Lingua-4D6D9A)
![Refinement](https://img.shields.io/badge/Refinement-Text%20%2B%20Timestamps-0EA5E9)

| Focus | Value |
| --- | --- |
| Input | FFmpeg-compatible audio/video |
| Pipeline | VAD segmentation → Whisper transcription → Lingua refinement |
| Output | Normalized `*.wav`, `*.srt`, y `*.json` |
| Best use | Subtítulos en varios idiomas con etiquetas de idioma por segmento |

---

## Índice

- [Resumen](#-resumen)
- [A un vistazo](#a-un-vistazo)
- [Características clave](#-características-clave)
- [Flujo del pipeline](#-flujo-del-pipeline)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Requisitos previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Inicio rápido](#-inicio-rápido)
- [Uso](#-uso)
- [Configuración](#️-configuración)
- [Formato de salida](#-formato-de-salida)
- [Ejemplos](#-ejemplos)
- [Notas de desarrollo](#-notas-de-desarrollo)
- [Solución de problemas](#-solución-de-problemas)
- [Limitaciones y suposiciones conocidas](#-limitaciones-y-suposiciones-conocidas)
- [Hoja de ruta](#-hoja-de-ruta)
- [Support](#-support)
- [Contacto](#-contacto)
- [Agradecimientos](#-agradecimientos)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## ✨ Resumen

`MultilingualWhisper` es un pipeline CLI en Python centrado en [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Combina:

- Silero VAD para segmentación de voz
- OpenAI Whisper para transcripción y predicción inicial de idioma
- Lingua para refinamiento de idioma basado en texto
- FFmpeg para extracción, normalización y manejo de medios

Las salidas principales son archivos de subtítulos en `.srt` y `.json`, además del audio `.wav` extraído y normalizado.

### A un vistazo

| Elemento | Detalles |
|---|---|
| Punto de entrada principal | `vad_lang_subtitle.py` |
| Entrada | Vídeo/audio compatible con FFmpeg |
| Salida | `*.wav`, `*.srt`, `*.json` |
| Flujo principal | VAD -> Whisper -> Lingua -> refinamiento |
| Caso de uso típico | Generación de subtítulos con múltiples idiomas |

---

## 🚀 Características clave

- **Pipeline Silero VAD -> Whisper**
  La detección de actividad de voz (VAD) divide el audio en segmentos de voz y luego Whisper transcribe cada fragmento.

- **Detección de idioma de alta granularidad**
  Usa [Lingua](https://github.com/pemistahl/lingua-java) junto con el detector propio de Whisper para etiquetar cada segmento (incluso palabras individuales) con códigos de idioma ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Refinamiento inteligente de segmentos**
  La limpieza de marcas de tiempo garantiza que no haya huecos ni solapamientos. Las divisiones por puntuación separan transcripciones largas por comas, puntos, signos de interrogación, etc. VAD vuelve a alinear las palabras con bloques de VAD para subtítulos más fluidos. La segmentación según longitud aplica límites específicos por idioma.

- **Subtítulos multilingües**
  Genera `.srt` y `.json`, preservando etiquetas de idioma por segmento para que puedas aplicar estilos o filtros por idioma en reproductores o editores de forma posterior.

- **Gestión robusta de medios**
  Extrae y normaliza audio automáticamente con FFmpeg, intenta reparar contenedores dañados y aplica normalización dinámica (`dynaudnorm`) para subtítulos más claros.

---

## 🔁 Flujo del pipeline

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
2. Resuelve las rutas de salida desde el nombre base de entrada.
3. Extrae/normaliza audio con FFmpeg.
4. Carga Silero VAD (`torch.hub`) y el modelo de Whisper.
5. Realiza la transcripción en dos pasos sobre los fragmentos VAD.
6. Fusiona/refina segmentos y luego hace una segunda transcripción sobre tramos fusionados.
7. Aplica reducción de longitud de subtítulos y limpieza de marcas de tiempo.
8. Guarda `.srt` y `.json`.

---

## 🗂 Estructura del proyecto

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

> ⚠️ Nota: En el README anterior se hacía referencia a `requirements.txt`, pero actualmente falta en la raíz del repositorio.

---

## ✅ Requisitos previos

- Python `3.10+` (probado con entornos modernos de Python 3.x)
- `ffmpeg` instalado y disponible en `PATH`
- CPU/GPU + RAM suficientes para el modelo Whisper elegido (para `large`, se recomienda encarecidamente GPU)
- Conexión a internet en la primera ejecución para descargar los pesos de Whisper y los recursos de Silero VAD (`torch.hub`)

Paquetes de Python usados por el script:

- `torch`
- `torchaudio`
- `whisper` (paquete Python de OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

Comandos de verificación rápida:

```bash
python --version
ffmpeg -version
```

---

## 🔧 Instalación

1. **Clona este repositorio**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **Crea y activa un entorno virtual**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Instala dependencias**

```bash
pip install -r requirements.txt
```

Si `requirements.txt` aún no está en tu checkout, instala manualmente las dependencias principales:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Y asegúrate de tener FFmpeg instalado a nivel del sistema.

---

## ⚡ Inicio rápido

Si quieres el camino más rápido desde el clon hasta los subtítulos:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Sugerencia: usa `small` durante la iteración y luego cambia a `large` para la calidad final.

Artefactos esperados junto a tu medio de entrada:

- `*.wav` audio extraído y normalizado
- `*.srt` archivo de subtítulos para reproductores/editores
- `*.json` metadatos estructurados de subtítulos multilingües

---

## 🛠 Uso

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Opciones CLI

| Flag | Alias | Requerido | Descripción |
|---|---|---|---|
| `--video-path` | `-t` | Sí | Ruta del medio de entrada (vídeo/audio compatible con FFmpeg) |
| `--whisper-model` | — | No | Nombre del modelo Whisper (predeterminado: `large`) |
| `--force` | — | No | Vuelve a ejecutar aunque ya existan `.wav`, `.srt` o `.json` |

### Comportamiento de procesamiento

- Los nombres de salida se derivan de la ruta base de entrada.
- Para `input.mp4`, las salidas son `input.wav` (audio normalizado), `input.srt` (subtítulos con marcas de tiempo) y `input.json` (metadatos incluyendo `start`, `end`, `lang`, `text`, y opcionalmente temporización de palabras).
- La existencia de `.srt` o `.json` provoca saltar el procesamiento salvo que se establezca `--force`.

---

## ⚙️ Configuración

La configuración actual se controla principalmente por CLI y por valores por defecto en el código:

| Área de configuración | Comportamiento actual |
|---|---|
| Modelo Whisper | `--whisper-model` (predeterminado `large`) |
| Frecuencia de muestreo de procesamiento | Fijada en `16000` para VAD/transcripción |
| Extracción FFmpeg | WAV mono, `44100 Hz`, con `dynaudnorm=f=100` |
| Detector Lingua | Inicializado para `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` en el flujo principal |
| Valores predeterminados de filtrado de Whisper | Incluye `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Nota de suposición: las listas de idiomas en los valores predeterminados del helper y en la configuración principal del detector no son idénticas por completo; este README conserva el comportamiento actual tal como está implementado.

---

## 📦 Formato de salida

La herramienta escribe dos artefactos de subtítulos por medio de entrada:

- `*.srt`: texto estándar de subtítulos con marcas de tiempo `HH:MM:SS,mmm`.
- `*.json`: lista estructurada de subtítulos que contiene marcas de tiempo formateadas y etiquetas de idioma.

Forma típica de segmento JSON:

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

- `start`/`end` se serializan como cadenas con formato SRT en la salida JSON.
- `words` puede estar presente según la etapa de procesamiento/refinamiento del segmento.
- Un valor `lang` de `und` puede aparecer en tramos con idioma incierto.

---

## 🧪 Ejemplos

Ejecutar en un MP4:

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Ejecutar en un MOV y forzar sobreescritura:

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Ejecutar en una entrada solo de audio compatible con FFmpeg:

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Ejemplo por lotes (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Notas de desarrollo

- El script activo canónico es `vad_lang_subtitle.py`.
- Los archivos históricos (`*.old`, `*.shorterlength*`, `archived/`) son útiles como referencia, pero parecen no canónicos.
- Actualmente no hay scaffolding de paquete de proyecto (`pyproject.toml`, `setup.py`) ni suite de CI/tests en el repositorio.
- `data/` contiene artefactos de medios grandes de muestra; ten en cuenta el tamaño del repositorio y el uso de disco local durante pruebas.
- `clean_subtitles_dict()` existe en el código, pero actualmente no se invoca desde el pipeline principal.
- `--force` es el mecanismo actual para garantizar la regeneración de salidas durante ajustes iterativos.

Sugerencia de ciclo local de desarrollo:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Usa un modelo más pequeño (`tiny`/`base`/`small`) durante la iteración y luego cambia a `large` para la calidad final.

---

## 🩺 Solución de problemas

| Síntoma | Qué hacer |
|---|---|
| `ffmpeg: command not found` | Instala FFmpeg y verifica con `ffmpeg -version`. |
| La primera ejecución es muy lenta o parece colgada | Las descargas iniciales de modelos (Whisper + Silero) pueden tardar; las ejecuciones posteriores suelen ser más rápidas. |
| Errores de CUDA / GPU | Prueba el fallback a CPU usando un modelo Whisper más pequeño (`small`, `base`, `tiny`) y asegúrate de tener una build de PyTorch compatible con tu entorno. |
| Los archivos de salida no se regeneran | Usa `--force` para sobrescribir archivos derivados existentes. |
| `pip install -r requirements.txt` falla porque no se encuentra el archivo | Usa el comando de instalación manual mostrado en Instalación. |
| Etiquetado de idioma impreciso en segmentos cortos | Puede ocurrir en segmentos extremadamente cortos o ruidosos; la lógica actual combina Whisper y Lingua pero aún tiene casos límite. |
| Salida de subtítulos vacía o casi vacía | Confirma que la entrada tiene voz, revisa el `.wav` extraído y vuelve a intentar con `--force` tras validar la extracción con FFmpeg. |
| Cambios de idioma inesperados entre líneas vecinas | Esto puede ocurrir en segmentos muy cortos; considera hacer post-unión en herramientas posteriores por idioma y duración mínima. |

---

## ⚠️ Limitaciones y suposiciones conocidas

- El manifiesto de dependencias no está incluido (`requirements.txt`, `pyproject.toml` y `setup.py` no están presentes en la raíz en el momento de esta redacción).
- La licencia se declara como MIT en el README, pero actualmente no existe un archivo `LICENSE` independiente.
- Lingua se inicializa explícitamente con `EN/ZH/JA/AR` en el flujo principal, mientras que los valores predeterminados del helper incluyen más códigos candidatos.
- No se han añadido pruebas automáticas/benchmarks, por lo que la validación es principalmente manual.
- Hay scripts históricos en la raíz y en `archived/`; solo `vad_lang_subtitle.py` debe tratarse como activo salvo que experimentes intencionalmente.

---

## 🗺 Hoja de ruta

- Añadir y mantener un `requirements.txt` o `pyproject.toml` versionado.
- Añadir pruebas automáticas para la lógica de segmentación y limpieza de marcas de tiempo.
- Añadir documentación de benchmark y evaluación de calidad para casos límite multilingües.
- Añadir soporte opcional de archivo de configuración en lugar de depender solo de valores por defecto en el código.
- Ampliar el conjunto de README en `i18n/` y mantener sincronizadas las barras de idiomas.
- Aclarar y unificar el comportamiento de selección de idioma entre la configuración del detector y los valores predeterminados del helper.
- Añadir un archivo `LICENSE` formal para que coincida con la declaración del README.

---

## 🔗 Agradecimientos

- [OpenAI Whisper](https://github.com/openai/whisper) por speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) por detección robusta de actividad de voz
- [Lingua](https://github.com/pemistahl/lingua-java) por identificación de idioma de alta precisión

---

## 🤝 Contribuciones

1. Haz un fork y clona el repositorio
2. Crea una rama: `git checkout -b feat/your-idea`
3. Haz commit y push
4. Abre un PR

Para cambios sustanciales, incluye:

- Una breve descripción del cambio de comportamiento esperado
- Un ejemplo de comando reproducible
- Fragmentos de subtítulos antes/después cuando aplique

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contacto

- Abre un issue para informes de errores, preguntas de uso y solicitudes de funciones.
- Usa las opciones de apoyo anteriores para consultas sobre patrocinio y donaciones.

## 📄 License

MIT © Lachlan Chen
