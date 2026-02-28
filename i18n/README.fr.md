[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


<p align="center">
  <img src="https://raw.githubusercontent.com/lachlanchen/lachlanchen/main/logos/banner.png" alt="Bannière LazyingArt" />
</p>

# MultilingualWhisper

Un générateur de sous-titres prêt à l’emploi, basé sur OpenAI Whisper, étendu avec une détection de langue précise par segment et un affinement pour les vidéos contenant des langues mixtes.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-black)
![VAD](https://img.shields.io/badge/VAD-Silero-green)
![Lang Detect](https://img.shields.io/badge/Language%20Detection-Lingua-2ea44f)
![FFmpeg](https://img.shields.io/badge/Media-FFmpeg-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Vue d’ensemble

`MultilingualWhisper` est un pipeline CLI Python centré sur [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Il combine :

- Silero VAD pour la segmentation de la parole
- OpenAI Whisper pour la transcription et la prédiction initiale de la langue
- Lingua pour l’affinage de la langue basé sur le texte
- FFmpeg pour l’extraction, la normalisation et la gestion des médias

Les sorties principales sont des fichiers de sous-titres `.srt` et `.json`, ainsi que des fichiers audio `.wav` extraits et normalisés.

### En bref

| Élément | Détails |
|---|---|
| Point d’entrée principal | `vad_lang_subtitle.py` |
| Entrée | Vidéo/audio pris en charge par FFmpeg |
| Sortie | `*.wav`, `*.srt`, `*.json` |
| Flux principal | VAD -> Whisper -> Lingua -> affinement |
| Cas d’usage typique | Génération de sous-titres multilingues |

---

## 🚀 Fonctionnalités clés

- **Pipeline Silero VAD -> Whisper**  
  La détection d’activité vocale (VAD) découpe l’audio en segments de parole, puis Whisper transcrit chaque bloc.

- **Détection de langue fine**  
  Utilise [Lingua](https://github.com/pemistahl/lingua-java) avec le détecteur natif de Whisper pour étiqueter chaque segment (voire chaque mot) avec des codes de langue ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Affinage intelligent des segments**  
  Le nettoyage des horodatages supprime les trous et chevauchements. Les découpes basées sur la ponctuation divisent les transcriptions longues aux virgules, points, points d’interrogation, etc. Les fusions VAD réalignent les mots sur les blocs VAD pour des sous-titres plus fluides. La segmentation sensible à la longueur applique des limites spécifiques à chaque langue.

- **Sous-titres multilingues**  
  Produit à la fois `.srt` et `.json`, en conservant les étiquettes de langue par segment afin de pouvoir styliser ou filtrer par langue dans les lecteurs/éditeurs en aval.

- **Gestion robuste des médias**  
  Extrait et normalise automatiquement l’audio via FFmpeg, tente de réparer les conteneurs endommagés et applique une normalisation dynamique (`dynaudnorm`) pour des transcriptions plus claires.

---

## 🗂 Structure du projet

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
├── i18n/                              # Translation/readme workspace
└── .auto-readme-work/                 # README generation workspace artifacts
```

> ⚠️ Note : un README précédent faisait référence à `requirements.txt`, mais ce fichier est actuellement absent à la racine du dépôt.

---

## ✅ Prérequis

- Python `3.10+` (testé avec des environnements 3.x récents)
- `ffmpeg` installé et accessible sur `PATH`
- CPU/GPU + RAM suffisants pour le modèle Whisper choisi (pour `large`, un GPU est fortement recommandé)
- Accès Internet au premier lancement pour récupérer les poids de modèle Whisper et les ressources Silero VAD (`torch.hub`)

Les paquets Python utilisés par le script incluent :

- `torch`
- `torchaudio`
- `whisper` (paquet Python OpenAI Whisper)
- `lingua-language-detector`
- `tqdm`

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

| Option | Alias | Requis | Description |
|---|---|---|---|
| `--video-path` | `-t` | Oui | Chemin du média d’entrée (vidéo/audio pris en charge par FFmpeg) |
| `--whisper-model` | — | Non | Nom du modèle Whisper (par défaut : `large`) |
| `--force` | — | Non | Relance le traitement même si `.wav`, `.srt` ou `.json` existent déjà |

### Comportement du traitement

- Les noms de sortie sont dérivés du chemin de base de l’entrée.
- Pour `input.mp4`, les sorties sont `input.wav` (audio normalisé), `input.srt` (sous-titres horodatés) et `input.json` (métadonnées incluant `start`, `end`, `lang`, `text`, et éventuellement les timings mot à mot).
- Si `.srt` ou `.json` existent déjà, le traitement est ignoré sauf si `--force` est défini.

---

## ⚙️ Configuration

La configuration actuelle est principalement pilotée par la CLI et les valeurs par défaut du code :

- Modèle Whisper : `--whisper-model` (par défaut `large`)
- Fréquence d’échantillonnage : fixée en dur à `16000` pour le traitement
- Extraction FFmpeg : WAV mono, `44100 Hz`, avec `dynaudnorm=f=100`
- Détecteur Lingua : initialisé pour `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` dans le flux principal
- Les codes langue autorisés pour le filtrage côté Whisper incluent `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` dans les valeurs par défaut des fonctions utilitaires

Note d’hypothèse : les listes de langues des valeurs par défaut utilitaires et de l’initialisation du détecteur principal ne sont pas totalement identiques ; ce README conserve le comportement actuel tel qu’implémenté.

---

## 🧪 Exemples

Exécution sur un MP4 :

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Exécution sur un MOV et forçage de l’écrasement :

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Exécution sur une entrée audio uniquement prise en charge par FFmpeg :

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

---

## 🧭 Notes de développement

- Le script canonique actif est `vad_lang_subtitle.py`.
- Les fichiers historiques (`*.old`, `*.shorterlength*`, `archived/`) sont utiles comme référence mais semblent non canoniques.
- Il n’y a actuellement ni structure de packaging (`pyproject.toml`, `setup.py`), ni suite CI/tests commitée.
- `data/` contient de gros artefacts média d’exemple ; faites attention à la taille du dépôt et à l’utilisation de disque local pendant les expérimentations.
- `clean_subtitles_dict()` existe dans le code mais n’est actuellement pas appelée par le pipeline principal.

---

## 🩺 Dépannage

| Symptôme | Que faire |
|---|---|
| `ffmpeg: command not found` | Installez FFmpeg et vérifiez avec `ffmpeg -version`. |
| Le premier lancement est très lent ou semble bloqué | Les téléchargements initiaux de modèles (Whisper + Silero) peuvent prendre du temps ; les exécutions suivantes sont plus rapides. |
| Erreurs CUDA / GPU | Essayez un repli CPU avec un modèle Whisper plus petit (`small`, `base`, `tiny`) et assurez-vous d’utiliser une version PyTorch adaptée à votre environnement. |
| Les fichiers de sortie ne sont pas régénérés | Utilisez `--force` pour écraser les fichiers dérivés existants. |
| `pip install -r requirements.txt` échoue car le fichier est introuvable | Utilisez la commande d’installation manuelle des dépendances indiquée dans Installation. |
| Étiquetage de langue imprécis sur des segments courts | Cela peut arriver sur des passages très courts/bruités ; la logique actuelle combine Whisper et Lingua mais comporte encore des cas limites. |

---

## 🗺 Feuille de route

- Ajouter et maintenir un `requirements.txt` ou `pyproject.toml` versionné.
- Ajouter des tests automatisés pour la segmentation et la logique de nettoyage des horodatages.
- Ajouter une documentation de benchmark et d’évaluation qualité pour les cas limites multilingues.
- Ajouter la prise en charge optionnelle d’un fichier de configuration au lieu d’un comportement uniquement basé sur les valeurs par défaut du code.
- Étendre l’ensemble des README i18n dans `i18n/` et garder les barres de langues synchronisées.

---

## 💖 Support

Si ce projet vous aide, vous pouvez soutenir son développement via :

- GitHub Sponsors: https://github.com/sponsors/lachlanchen
- Site personnel: https://lazying.art
- Chat/communauté: https://chat.lazying.art
- Hub d’idées/projets: https://onlyideas.art

---

## 🔗 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) pour la reconnaissance vocale
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) pour une détection robuste de l’activité vocale
- [Lingua](https://github.com/pemistahl/lingua-java) pour l’identification de langue à haute précision

---

## 🤝 Contribuer

1. Forkez et clonez
2. Créez une branche : `git checkout -b feat/your-idea`
3. Committez et poussez
4. Ouvrez une PR

---

## 📄 Licence

MIT © Lachlan Chen
