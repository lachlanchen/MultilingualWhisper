[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="LazyingArt banner" />
</p>

# MultilingualWhisper

Un générateur de sous-titres prêt à l’emploi basé sur OpenAI Whisper, enrichi d’une détection de langue précise par segment et d’un raffinage pour les vidéos contenant des langues mixtes.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-111111)
![VAD](https://img.shields.io/badge/VAD-Silero-2EA44F)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-0E8A16)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-FF6F00?logo=ffmpeg&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Table des matières

- [Vue d’ensemble](#-vue-densemble)
- [En bref](#en-bref)
- [Fonctionnalités clés](#-fonctionnalités-clés)
- [Flux du pipeline](#-flux-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Format de sortie](#-format-de-sortie)
- [Exemples](#-exemples)
- [Notes de développement](#-notes-de-développement)
- [Dépannage](#-dépannage)
- [Limites connues et hypothèses](#-limites-connues-et-hypothèses)
- [Feuille de route](#-feuille-de-route)
- [Support](#-support)
- [Remerciements](#-remerciements)
- [Contribuer](#-contribuer)
- [Licence](#-licence)

---

## ✨ Vue d’ensemble

`MultilingualWhisper` est un pipeline CLI Python centré sur [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Il combine :

- Silero VAD pour la segmentation de la parole
- OpenAI Whisper pour la transcription et la prédiction initiale de langue
- Lingua pour l’affinage de la langue basé sur le texte
- FFmpeg pour l’extraction, la normalisation et la gestion des médias

Les sorties principales sont des fichiers de sous-titres `.srt` et `.json`, ainsi qu’un audio `.wav` extrait et normalisé.

### En bref

| Élément | Détails |
|---|---|
| Point d’entrée principal | `vad_lang_subtitle.py` |
| Entrée | Vidéo/audio pris en charge par FFmpeg |
| Sortie | `*.wav`, `*.srt`, `*.json` |
| Flux principal | VAD -> Whisper -> Lingua -> raffinage |
| Cas d’usage typique | Génération de sous-titres multilingues |

---

## 🚀 Fonctionnalités clés

- **Pipeline Silero VAD -> Whisper**
  La détection d’activité vocale (VAD) découpe l’audio en segments de parole, puis Whisper transcrit chaque portion.

- **Détection de langue fine**
  Utilise [Lingua](https://github.com/pemistahl/lingua-java) avec le détecteur natif de Whisper pour étiqueter chaque segment (voire chaque mot) avec des codes ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Raffinage intelligent des segments**
  Le nettoyage des timestamps garantit l’absence de trous ou de chevauchements. Les découpes sur ponctuation cassent les transcriptions longues aux virgules, points, points d’interrogation, etc. Les fusions VAD réalignent les mots sur les blocs VAD pour des sous-titres plus fluides. La segmentation sensible à la longueur applique des limites spécifiques à chaque langue.

- **Sous-titres multilingues**
  Produit à la fois du `.srt` et du `.json`, en conservant les balises de langue par segment pour permettre le style ou le filtrage par langue dans les lecteurs/éditeurs en aval.

- **Gestion robuste des médias**
  Extrait et normalise automatiquement l’audio via FFmpeg, tente de réparer les conteneurs cassés, et applique une normalisation dynamique (`dynaudnorm`) pour des transcriptions plus claires.

---

## 🔁 Flux du pipeline

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

Chemin d’exécution principal dans `vad_lang_subtitle.py` :

1. Analyse des arguments CLI (`--video-path`, `--whisper-model`, `--force`).
2. Résolution des chemins de sortie à partir du nom de base d’entrée.
3. Extraction/normalisation audio via FFmpeg.
4. Chargement de Silero VAD (`torch.hub`) et du modèle Whisper.
5. Première passe de transcription sur les blocs VAD.
6. Fusion/raffinage des segments, puis seconde passe de transcription sur les plages fusionnées.
7. Application de la réduction de longueur des sous-titres et du nettoyage des timestamps.
8. Sauvegarde des fichiers `.srt` et `.json`.

---

## 🗂 Structure du projet

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

> ⚠️ Note : Le README précédent faisait référence à `requirements.txt`, mais ce fichier est actuellement absent à la racine du dépôt.

---

## ✅ Prérequis

- Python `3.10+` (testé avec des environnements 3.x récents)
- `ffmpeg` installé et disponible dans le `PATH`
- CPU/GPU + RAM suffisants pour le modèle Whisper choisi (pour `large`, un GPU est fortement recommandé)
- Accès Internet au premier lancement pour récupérer les poids Whisper et les ressources Silero VAD (`torch.hub`)

Les paquets Python utilisés par le script incluent :

- `torch`
- `torchaudio`
- `whisper` (paquet Python OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

Commandes de vérification rapide :

```bash
python --version
ffmpeg -version
```

---

## 🔧 Installation

1. **Cloner ce dépôt**

```bash
git clone git@github.com:lachlanchen/MultilingualWhisper.git
cd MultilingualWhisper
```

2. **Créer et activer un environnement virtuel**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

Si `requirements.txt` est toujours absent de votre copie locale, installez manuellement les dépendances d’exécution principales :

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Et assurez-vous que FFmpeg est installé au niveau système.

---

## 🛠 Utilisation

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Options CLI

| Flag | Alias | Obligatoire | Description |
|---|---|---|---|
| `--video-path` | `-t` | Oui | Chemin du média d’entrée (vidéo/audio pris en charge par FFmpeg) |
| `--whisper-model` | — | Non | Nom du modèle Whisper (par défaut : `large`) |
| `--force` | — | Non | Relance même si `.wav`, `.srt` ou `.json` existent déjà |

### Comportement du traitement

- Les noms de sortie sont dérivés du chemin de base en entrée.
- Pour `input.mp4`, les sorties sont `input.wav` (audio normalisé), `input.srt` (sous-titres horodatés), et `input.json` (métadonnées incluant `start`, `end`, `lang`, `text`, et éventuellement les timings de mots).
- La présence d’un `.srt` ou `.json` existant provoque un saut sauf si `--force` est défini.

---

## ⚙️ Configuration

La configuration actuelle est principalement pilotée par la CLI et les valeurs par défaut du code :

| Zone de config | Comportement actuel |
|---|---|
| Modèle Whisper | `--whisper-model` (par défaut `large`) |
| Taux d’échantillonnage de traitement | Codé en dur à `16000` pour le traitement VAD/transcription |
| Extraction FFmpeg | WAV mono, `44100 Hz`, avec `dynaudnorm=f=100` |
| Détecteur Lingua | Initialisé pour `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` dans le flux principal |
| Valeurs par défaut helper côté filtrage Whisper | Inclut `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Note d’hypothèse : les listes de langues des valeurs helper par défaut et la configuration du détecteur principal ne sont pas totalement identiques ; ce README conserve le comportement actuellement implémenté.

---

## 📦 Format de sortie

L’outil écrit deux artefacts de sous-titres par média d’entrée :

- `*.srt` : texte de sous-titres standard avec timestamps `HH:MM:SS,mmm`.
- `*.json` : liste de sous-titres structurée contenant timestamps formatés et balises de langue.

Forme typique d’un segment JSON :

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

Notes :

- `start`/`end` sont sérialisés en chaînes au format SRT dans la sortie JSON.
- `words` peut être présent selon l’étape de traitement/raffinage du segment.
- Une valeur `lang` égale à `und` peut apparaître pour des segments dont la langue est incertaine.

---

## 🧪 Exemples

Exécution sur un MP4 :

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Exécution sur un MOV avec écrasement forcé :

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Exécution sur une entrée audio seule prise en charge par FFmpeg :

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Exemple batch shell (bash) :

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Notes de développement

- Le script actif canonique est `vad_lang_subtitle.py`.
- Les fichiers historiques (`*.old`, `*.shorterlength*`, `archived/`) sont utiles comme référence, mais semblent non canoniques.
- Il n’existe actuellement ni scaffolding de projet packagé (`pyproject.toml`, `setup.py`) ni suite CI/tests commitée.
- `data/` contient de gros artefacts média d’exemple ; surveillez la taille du dépôt et l’espace disque local pendant les expérimentations.
- `clean_subtitles_dict()` existe dans le code mais n’est actuellement pas invoqué par le pipeline principal.
- `--force` est le mécanisme actuel pour garantir la régénération des sorties lors des itérations de réglage.

Boucle de dev locale suggérée :

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Utilisez un modèle plus petit (`tiny`/`base`/`small`) pendant les itérations, puis passez à `large` pour la qualité finale.

---

## 🩺 Dépannage

| Symptôme | Que faire |
|---|---|
| `ffmpeg: command not found` | Installez FFmpeg et vérifiez avec `ffmpeg -version`. |
| Le premier lancement est très lent ou semble bloqué | Les téléchargements initiaux des modèles (Whisper + Silero) peuvent prendre du temps ; les relances sont plus rapides. |
| Erreurs CUDA / GPU | Essayez un repli CPU avec un plus petit modèle Whisper (`small`, `base`, `tiny`) et assurez-vous d’avoir une build PyTorch adaptée à votre environnement. |
| Les fichiers de sortie ne sont pas régénérés | Utilisez `--force` pour écraser les fichiers dérivés existants. |
| `pip install -r requirements.txt` échoue car le fichier est introuvable | Utilisez la commande d’installation manuelle des dépendances indiquée dans Installation. |
| Étiquetage de langue imprécis sur des segments courts | Cela peut arriver sur des segments très courts/bruités ; la logique actuelle combine Whisper et Lingua mais garde des cas limites. |
| Sortie de sous-titres vide ou quasi vide | Vérifiez que l’entrée contient de la parole, inspectez le `.wav` extrait, puis relancez avec `--force` après validation de l’extraction FFmpeg. |
| Bascules de langue inattendues entre lignes voisines | Cela peut survenir sur des segments très courts ; envisagez une fusion post-traitement en aval par langue et durée minimale. |

---

## ⚠️ Limites connues et hypothèses

- Le manifeste de dépendances n’est pas commité (`requirements.txt`, `pyproject.toml`, et `setup.py` sont absents de la racine du dépôt au moment de la rédaction).
- La licence est déclarée MIT dans le README, mais un fichier `LICENSE` autonome n’est actuellement pas présent.
- Lingua est explicitement initialisé avec `EN/ZH/JA/AR` dans le flux principal, tandis que les valeurs helper par défaut incluent plus de codes candidats.
- Aucun test/benchmark automatisé n’est actuellement commité ; la validation est donc principalement manuelle.
- Des scripts historiques sont présents à la racine et dans `archived/` ; seul `vad_lang_subtitle.py` doit être considéré comme actif sauf expérimentation volontaire.

---

## 🗺 Feuille de route

- Ajouter et maintenir un `requirements.txt` ou un `pyproject.toml` versionné.
- Ajouter des tests automatisés pour la segmentation et le nettoyage des timestamps.
- Ajouter une documentation d’évaluation benchmark/qualité pour les cas limites multilingues.
- Ajouter la prise en charge optionnelle d’un fichier de config au lieu d’un comportement uniquement basé sur des valeurs par défaut codées.
- Étendre l’ensemble des README i18n dans `i18n/` et garder les barres de langue synchronisées.
- Clarifier et unifier le comportement de sélection de langue entre la configuration du détecteur et les valeurs helper par défaut.
- Ajouter un fichier `LICENSE` formel pour correspondre à la déclaration du README.

---

## 💖 Support

Si ce projet vous aide, vous pouvez soutenir son développement via :

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Site personnel : https://lazying.art
- Chat/communauté : https://chat.lazying.art
- Hub idées/projets : https://onlyideas.art

---

## 🔗 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) pour la reconnaissance vocale
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) pour une détection d’activité vocale robuste
- [Lingua](https://github.com/pemistahl/lingua-java) pour l’identification de langue haute précision

---

## 🤝 Contribuer

1. Forkez et clonez
2. Créez une branche : `git checkout -b feat/your-idea`
3. Commit et push
4. Ouvrez une PR

Pour les changements substantiels, incluez :

- Une courte description du changement de comportement attendu
- Un exemple de commande reproductible
- Des extraits de sous-titres avant/après quand pertinent

---

## 📄 Licence

MIT © Lachlan Chen
