[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Un generador de subtítulos listo para usar basado en OpenAI Whisper, ampliado con detección y refinamiento precisos del idioma por segmento para videos que contienen idiomas mixtos.

> Genera subtítulos multilingües más limpios a partir de contenido real con idiomas mezclados mediante segmentación consciente del idioma.

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
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-6B7280)
![Maintained](https://img.shields.io/badge/Maintained-Yes-16A34A)

> 🌍 **Documentación multilingüe disponible**: inglés + 10 variantes traducidas del README en [`i18n/`](../i18n/), enlazadas en la barra de idiomas de arriba.

### Idiomas de la documentación

| Locale | Archivo |
| --- | --- |

| Enfoque | Valor |
| --- | --- |
| Entrada | Audio/video compatible con FFmpeg |
| Flujo | Segmentación VAD -> Transcripción Whisper -> Refinamiento con Lingua |
| Salida | `*.wav`, `*.srt` y `*.json` normalizados |
| Mejor uso | Subtítulos en contenido multilingüe con etiquetas de idioma por segmento |

---

## Table of Contents

- [Resumen](#-resumen)
- [De un Vistazo](#de-un-vistazo)
- [Características Clave](#-características-clave)
- [Flujo de Pipeline](#-flujo-de-pipeline)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
- [Inicio Rápido](#-inicio-rápido)
- [Guía de Selección de Modelo](#-guía-de-selección-de-modelo)
- [Uso](#-uso)
- [Configuración](#-configuración)
- [Formato de Salida](#-formato-de-salida)
- [Ejemplos](#-ejemplos)
- [Notas de Desarrollo](#-notas-de-desarrollo)
- [Solución de Problemas](#-solución-de-problemas)
- [Limitaciones y Suposiciones Conocidas](#-limitaciones-y-suposiciones-conocidas)
- [Hoja de Ruta](#-hoja-de-ruta)
- [Agradecimientos](#-agradecimientos)
- [Contribuir](#-contribuir)
- [Support](#-support)
- [Contacto](#-contacto)
- [Licencia](#-licencia)

---

## ✨ Resumen

`MultilingualWhisper` es un pipeline CLI de Python centrado en [`vad_lang_subtitle.py`](../vad_lang_subtitle.py). Combina:

- Silero VAD para segmentación de voz
- OpenAI Whisper para transcripción y predicción inicial del idioma
- Lingua para refinamiento del idioma basado en texto
- FFmpeg para extracción, normalización y manejo de medios

Las salidas principales son archivos de subtítulos en `.srt` y `.json`, además de audio `.wav` normalizado y extraído.

### De un Vistazo

| Elemento | Detalles |
|---|---|
| Punto de entrada principal | `vad_lang_subtitle.py` |
| Entrada | Video/audio compatible con FFmpeg |
| Salida | `*.wav`, `*.srt`, `*.json` |
| Flujo principal | VAD -> Whisper -> Lingua -> refinamiento |
| Caso de uso típico | Generación de subtítulos en contenido con idiomas mixtos |

---

## 🚀 Características Clave

- **Pipeline Silero VAD -> Whisper**  
  Voice Activity Detection (VAD) divide el audio en segmentos de voz y luego Whisper transcribe cada fragmento.

- **Detección de idioma de grano fino**  
  Usa [Lingua](https://github.com/pemistahl/lingua-java) junto con el detector nativo de Whisper para etiquetar cada segmento (incluso palabras individuales) con códigos de idioma ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Refinamiento inteligente de segmentos**  
  La limpieza de timestamps evita huecos y solapamientos. La división por puntuación parte transcripciones largas en comas, puntos, signos de interrogación, etc. El re-merge de VAD realinea palabras de vuelta a bloques VAD para subtítulos más fluidos. La segmentación consciente de longitud aplica límites específicos por idioma.

- **Subtítulos multilingües**  
  Genera `.srt` y `.json`, preservando etiquetas de idioma por segmento para que puedas dar estilo o filtrar por idioma en reproductores o editores posteriores.

- **Manejo robusto de medios**  
  Extrae y normaliza audio automáticamente con FFmpeg, intenta reparar contenedores dañados y aplica normalización dinámica (`dynaudnorm`) para obtener transcripciones más claras.

---

## 🔁 Flujo de Pipeline

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

1. Analizar argumentos CLI (`--video-path`, `--whisper-model`, `--force`).
2. Resolver rutas de salida a partir del basename de la entrada.
3. Extraer/normalizar audio mediante FFmpeg.
4. Cargar Silero VAD (`torch.hub`) y el modelo Whisper.
5. Realizar una primera transcripción sobre fragmentos VAD.
6. Unir/refinar segmentos y luego hacer una segunda transcripción sobre tramos fusionados.
7. Aplicar reducción de longitud de subtítulos y limpieza de timestamps.
8. Guardar `.srt` y `.json`.

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

> ⚠️ Nota: En versiones anteriores del README se referenciaba `requirements.txt`, pero actualmente no existe en la raíz del repositorio.

---

## ✅ Requisitos Previos

- Python `3.10+` (probado con entornos modernos de Python 3.x)
- `ffmpeg` instalado y disponible en `PATH`
- CPU/GPU + RAM suficientes para el modelo Whisper seleccionado (para `large`, se recomienda GPU con fuerza)
- Acceso a Internet en la primera ejecución para descargar pesos de Whisper y recursos de Silero VAD (`torch.hub`)

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

Si `requirements.txt` sigue ausente en tu checkout, instala manualmente las dependencias base de ejecución:

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Y asegúrate de que FFmpeg esté instalado a nivel de sistema.

---

## ⚡ Inicio Rápido

Si quieres la ruta más rápida desde `clone` hasta subtítulos:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Consejo: usa `small` mientras iteras y luego cambia a `large` para la calidad final.

Artefactos esperados junto a tu archivo de entrada:

- `*.wav` audio extraído y normalizado
- `*.srt` archivo de subtítulos para reproductores/editores
- `*.json` metadatos estructurados de subtítulos multilingües

---

## 🎚 Guía de Selección de Modelo

Elige un modelo Whisper en función de tu equilibrio entre velocidad y calidad:

| Modelo | Velocidad | Calidad | Uso recomendado |
|---|---|---|---|
| `tiny` / `base` | Más rápida | Más baja | Smoke tests rápidos y validación del pipeline |
| `small` | Rápida | Buena | Iteración diaria y desarrollo local |
| `medium` | Media | Mejor | Flujos de producción equilibrados |
| `large` (default) | Más lenta | Mejor | Exportaciones finales de subtítulos con máxima calidad |

Patrón práctico:

1. Itera con `small --force`
2. Valida timing y etiquetas de idioma
3. Reejecuta con `large --force` para la salida de entrega

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
| `--whisper-model` | — | No | Nombre del modelo Whisper (default: `large`) |
| `--force` | — | No | Reejecuta incluso si ya existen `.wav`, `.srt` o `.json` |

### Comportamiento del Procesamiento

- Los nombres de salida se derivan de la ruta base de la entrada.
- Para `input.mp4`, las salidas son `input.wav` (audio normalizado), `input.srt` (subtítulos con timestamps) y `input.json` (metadatos que incluyen `start`, `end`, `lang`, `text` y, opcionalmente, tiempos por palabra).
- Si ya existen `.srt` o `.json`, se omiten salvo que se especifique `--force`.

---

## ⚙️ Configuración

La configuración actual depende principalmente de CLI y de valores por defecto en código:

| Área de configuración | Comportamiento actual |
|---|---|
| Modelo Whisper | `--whisper-model` (default `large`) |
| Sample rate de procesamiento | Fijado en `16000` para el procesamiento de VAD/transcripción |
| Extracción con FFmpeg | WAV mono, `44100 Hz`, con `dynaudnorm=f=100` |
| Detector Lingua | Inicializado para `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` en el flujo principal |
| Defaults de helper de filtrado en Whisper | Incluye `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Nota sobre suposición: las listas de idiomas en los defaults del helper y la configuración principal del detector no son totalmente idénticas; este README preserva el comportamiento actual tal como está implementado.

Detalle adicional de implementación del script actual:

- `torch.set_num_threads(1)` se aplica en tiempo de ejecución.
- El modelo VAD se carga desde `snakers4/silero-vad` mediante `torch.hub.load(...)`.
- La limpieza de segmentos elimina entradas cuyo idioma es `und` o cuyo texto está vacío.

---

## 📦 Formato de Salida

La herramienta escribe dos artefactos de subtítulos por cada medio de entrada:

- `*.srt`: texto de subtítulos estándar con timestamps `HH:MM:SS,mmm`.
- `*.json`: lista estructurada de subtítulos con timestamps formateados y etiquetas de idioma.

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
- Un valor `lang` de `und` puede aparecer en tramos de idioma incierto.

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

Ejemplo de lote en shell (bash):

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Notas de Desarrollo

- El script canónico activo es `vad_lang_subtitle.py`.
- Los archivos históricos (`*.old`, `*.shorterlength*`, `archived/`) son útiles como referencia, pero parecen no canónicos.
- Actualmente no hay scaffolding de paquete (`pyproject.toml`, `setup.py`) ni suite de CI/tests confirmada.
- `data/` contiene artefactos de medios de ejemplo grandes; ten en cuenta el tamaño del repositorio y el uso de disco local durante los experimentos.
- `clean_subtitles_dict()` existe en el código, pero actualmente no se invoca desde el pipeline principal.
- `--force` es el mecanismo actual para garantizar la regeneración de salidas durante el ajuste iterativo.

Bucle local de desarrollo sugerido:

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Usa un modelo más pequeño (`tiny`/`base`/`small`) mientras iteras y luego cambia a `large` para la calidad final de salida.

---

## 🩺 Solución de Problemas

| Síntoma | Qué hacer |
|---|---|
| `ffmpeg: command not found` | Instala FFmpeg y verifica con `ffmpeg -version`. |
| La primera ejecución es muy lenta o parece bloqueada | Las descargas iniciales de modelos (Whisper + Silero) pueden tardar; las reejecuciones son más rápidas. |
| Errores de CUDA / GPU | Prueba fallback a CPU usando un modelo Whisper más pequeño (`small`, `base`, `tiny`) y asegúrate de usar una build de PyTorch compatible con tu entorno. |
| Los archivos de salida no se regeneran | Usa `--force` para sobrescribir archivos derivados existentes. |
| `pip install -r requirements.txt` falla porque no se encuentra el archivo | Usa el comando de instalación manual de dependencias mostrado en Instalación. |
| Etiquetado de idioma impreciso en segmentos cortos | Puede ocurrir en tramos extremadamente cortos/ruidosos; la lógica actual combina Whisper y Lingua, pero aún hay casos límite. |
| Salida de subtítulos vacía o casi vacía | Confirma que la entrada tenga voz, inspecciona el `.wav` extraído y vuelve a intentar con `--force` tras validar la extracción con FFmpeg. |
| Cambios inesperados de idioma entre líneas vecinas | Puede ocurrir en segmentos muy cortos; considera hacer post-merge por idioma y duración mínima en herramientas posteriores. |
| La extracción con FFmpeg falla en medios dañados | El script reintenta tras reparación de contenedor (`-c copy -movflags +faststart`), pero archivos muy corruptos pueden seguir fallando. |

Diagnóstico rápido:

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Limitaciones y Suposiciones Conocidas

- El manifiesto de dependencias no está versionado (`requirements.txt`, `pyproject.toml` y `setup.py` están ausentes en la raíz del repositorio al momento de escribir esto).
- La licencia se declara en el README como MIT, pero actualmente no hay un archivo `LICENSE` independiente.
- Lingua se inicializa explícitamente con `EN/ZH/JA/AR` en el flujo principal, mientras que los defaults del helper incluyen más códigos candidatos.
- Actualmente no hay tests/benchmarks automatizados versionados, por lo que la validación es principalmente manual.
- Hay scripts históricos en la raíz y en `archived/`; solo `vad_lang_subtitle.py` debe considerarse activo salvo que se esté experimentando intencionalmente.
- El script actualmente imprime logs verbosos en runtime y salida de depuración por segmento; es el comportamiento esperado en la implementación actual.

---

## 🗺 Hoja de Ruta

- Añadir y mantener un `requirements.txt` o `pyproject.toml` con versiones fijadas.
- Añadir tests automatizados para la lógica de segmentación y limpieza de timestamps.
- Añadir documentación de benchmarks y evaluación de calidad para casos límite multilingües.
- Añadir soporte opcional para archivo de configuración en lugar de solo defaults de código.
- Ampliar el set de README i18n en `i18n/` y mantener sincronizadas las barras de idioma.
- Aclarar y unificar el comportamiento de selección de idioma entre la configuración del detector y los defaults del helper.
- Añadir un archivo `LICENSE` formal para alinear con la declaración del README.

---

## 🔗 Agradecimientos

- [OpenAI Whisper](https://github.com/openai/whisper) por speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) por detección de actividad de voz robusta
- [Lingua](https://github.com/pemistahl/lingua-java) por identificación de idioma de alta precisión

---

## 🤝 Contribuir

1. Haz fork y clona
2. Crea una rama: `git checkout -b feat/your-idea`
3. Haz commit y push
4. Abre un PR

Para cambios sustanciales, incluye:

- Una descripción corta del cambio de comportamiento esperado
- Un ejemplo de comando reproducible
- Fragmentos de subtítulos antes/después cuando corresponda

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contacto

- Abre un issue para reportes de bugs, preguntas de uso y solicitudes de funcionalidades.
- Usa las opciones de soporte de arriba para consultas de patrocinio y donaciones.

---

## 📄 Licencia

MIT © Lachlan Chen
