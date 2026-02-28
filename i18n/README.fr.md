[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)



[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Un générateur de sous-titres prêt à l'emploi basé sur OpenAI Whisper, enrichi d'une détection de langue précise par segment et d'un affinage pour les vidéos contenant plusieurs langues.

> Générez des sous-titres multilingues plus propres à partir de médias réels en langues mixtes grâce à une segmentation sensible à la langue.

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
| Output | Normalized `*.wav`, `*.srt`, and `*.json` |
| Best use | Mixed-language subtitles with per-segment language tags |

---

## Table des matières

- [Aperçu](#-aperçu)
- [En bref](#en-bref)
- [Fonctionnalités clés](#-fonctionnalités-clés)
- [Flux du pipeline](#-flux-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Démarrage rapide](#-démarrage-rapide)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Format de sortie](#-format-de-sortie)
- [Exemples](#-exemples)
- [Notes de développement](#-notes-de-développement)
- [Dépannage](#-dépannage)
- [Limitations connues et hypothèses](#-limitations-connues-et-hypothèses)
- [Feuille de route](#-feuille-de-route)
- [Remerciements](#-remerciements)
- [Contribuer](#-contribuer)
- [Support](#-support)
- [Contact](#-contact)
- [Licence](#-licence)

---

## ✨ Aperçu

`MultilingualWhisper` est un pipeline CLI Python centré sur [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Il combine :

- Silero VAD pour la segmentation de la parole
- OpenAI Whisper pour la transcription et la prédiction initiale de la langue
- Lingua pour l'affinage de la langue basé sur le texte
- FFmpeg pour l'extraction, la normalisation et la manipulation des médias

Les sorties principales sont des fichiers de sous-titres en `.srt` et `.json`, ainsi que l'audio `.wav` extrait et normalisé.

### En bref

| Élément | Détails |
|---|---|
| Point d'entrée principal | `vad_lang_subtitle.py` |
| Entrée | Vidéo/audio pris en charge par FFmpeg |
| Sortie | `*.wav`, `*.srt`, `*.json` |
| Flux principal | VAD -> Whisper -> Lingua -> affinage |
| Cas d'usage typique | Génération de sous-titres multilingues |

---

## 🚀 Fonctionnalités clés

- **Pipeline Silero VAD -> Whisper**
  La détection d'activité vocale (VAD) découpe l'audio en segments de parole, puis Whisper transcrit chaque segment.

- **Détection précise de la langue**
  Utilise [Lingua](https://github.com/pemistahl/lingua-java) en complément du détecteur propre à Whisper pour étiqueter chaque segment (voire chaque mot) avec des codes ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Affinage intelligent des segments**
  Le nettoyage des timestamps évite les trous et chevauchements. Les séparations par ponctuation découpent les longues transcriptions aux virgules, points, points d'interrogation, etc. Les fusions VAD réalignent les mots sur les blocs VAD pour des sous-titres plus fluides. La segmentation sensible à la longueur applique des limites spécifiques par langue.

- **Sous-titres multilingues**
  Produit `.srt` et `.json`, en conservant les tags de langue par segment pour pouvoir styliser ou filtrer par langue dans les lecteurs ou éditeurs en aval.

- **Gestion robuste des médias**
  Extrait et normalise automatiquement l'audio via FFmpeg, tente de réparer les conteneurs endommagés et applique une normalisation dynamique (`dynaudnorm`) pour des transcriptions plus claires.

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

Chemin d'exécution principal dans `vad_lang_subtitle.py` :

1. Analyse des arguments CLI (`--video-path`, `--whisper-model`, `--force`).
2. Résolution des chemins de sortie depuis le nom de base d'entrée.
3. Extraction/normalisation audio via FFmpeg.
4. Chargement de Silero VAD (`torch.hub`) et du modèle Whisper.
5. Première passe de transcription sur les segments VAD.
6. Fusion/affinage des segments, puis deuxième passe de transcription sur les plages fusionnées.
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

> ⚠️ Note : Le README précédent faisait référence à `requirements.txt`, mais il est actuellement absent à la racine du dépôt.

---

## ✅ Prérequis

- Python `3.10+` (testé avec des environnements Python 3.x récents)
- `ffmpeg` installé et accessible via le `PATH`
- Ressources CPU/GPU + RAM suffisantes pour le modèle Whisper choisi (pour `large`, le GPU est fortement recommandé)
- Accès Internet au premier lancement pour télécharger les poids du modèle Whisper et les assets Silero VAD (`torch.hub`)

Les paquets Python utilisés par le script incluent :

- `torch`
- `torchaudio`
- `whisper` (package Python OpenAI Whisper)
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
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
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

Si `requirements.txt` est toujours absent de votre clone, installez manuellement les dépendances essentielles :

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Et vérifiez que FFmpeg est bien installé au niveau système.

---

## ⚡ Démarrage rapide

Si vous voulez le chemin le plus court entre le clone et les sous-titres :

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Astuce : utilisez `small` pendant vos essais, puis passez à `large` pour la qualité finale.

Artefacts attendus à côté de votre média d'entrée :

- `*.wav` audio extrait et normalisé
- `*.srt` fichier de sous-titres pour les lecteurs/éditeurs
- `*.json` métadonnées structurées de sous-titres multilingues

---

## 🛠 Utilisation

```bash
python vad_lang_subtitle.py \
  --video-path path/to/video.mp4 \
  --whisper-model large \
  [--force]
```

### Options CLI

| Drapeau | Alias | Obligatoire | Description |
|---|---|---|---|
| `--video-path` | `-t` | Oui | Chemin du média d'entrée (vidéo/audio pris en charge par FFmpeg) |
| `--whisper-model` | — | Non | Nom du modèle Whisper (par défaut : `large`) |
| `--force` | — | Non | Relance la génération même si `.wav`, `.srt` ou `.json` existent déjà |

### Comportement du traitement

- Les noms de sortie sont dérivés du chemin de base d'entrée.
- Pour `input.mp4`, les sorties sont `input.wav` (audio normalisé), `input.srt` (sous-titres horodatés) et `input.json` (métadonnées comprenant `start`, `end`, `lang`, `text`, avec éventuellement les timings de mots).
- La présence d'un `.srt` ou `.json` existant provoque un passage en mode ignoré, sauf si `--force` est renseigné.

---

## ⚙️ Configuration

La configuration actuelle est principalement pilotée par la CLI et les valeurs par défaut du code :

| Zone de configuration | Comportement actuel |
|---|---|
| Modèle Whisper | `--whisper-model` (par défaut `large`) |
| Taux d'échantillonnage de traitement | Défini en dur à `16000` pour VAD et transcription |
| Extraction FFmpeg | WAV mono, `44100 Hz`, avec `dynaudnorm=f=100` |
| Détecteur Lingua | Initialisé pour `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` dans le flux principal |
| Valeurs par défaut de filtrage Whisper | Inclut `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Point d'hypothèse : les listes de langues des paramètres par défaut du helper et de la configuration principale du détecteur ne sont pas strictement identiques ; ce README reflète le comportement actuel tel qu'implémenté.

---

## 📦 Format de sortie

L'outil produit deux artefacts de sous-titres par média d'entrée :

- `*.srt` : texte de sous-titres standard avec timestamps `HH:MM:SS,mmm`.
- `*.json` : liste structurée de sous-titres contenant des timestamps formatés et des tags de langue.

Forme typique d'un segment JSON :

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
- `words` peut être présent selon l'étape de traitement/affinage du segment.
- Une valeur `lang` de `und` peut apparaître pour des portions dont la langue est incertaine.

---

## 🧪 Exemples

Exécuter sur un MP4 :

```bash
python vad_lang_subtitle.py -t data/9b7bfbfbe8ab1b9925cfdc34f2f9f7_2024_03_15_22_08_26_COMPLETED.MP4 --whisper-model large
```

Exécuter sur un MOV en forçant l'écrasement :

```bash
python vad_lang_subtitle.py -t data/IMG_6276.MOV --whisper-model large --force
```

Exécuter sur un fichier audio seul pris en charge par FFmpeg :

```bash
python vad_lang_subtitle.py -t "data/深圳动物园中心喷泉.m4a" --whisper-model medium
```

Exemple batch (bash) :

```bash
for f in data/*.{MP4,MOV,m4a}; do
  [ -e "$f" ] || continue
  python vad_lang_subtitle.py -t "$f" --whisper-model medium
done
```

---

## 🧭 Notes de développement

- Le script actif canonique est `vad_lang_subtitle.py`.
- Les fichiers historiques (`*.old`, `*.shorterlength*`, `archived/`) sont utiles pour référence, mais semblent non canoniques.
- Il n'existe actuellement ni structure de packaging du projet (`pyproject.toml`, `setup.py`) ni suite CI/tests commitée.
- `data/` contient de grands médias d'exemple ; faites attention à la taille du dépôt et à l'espace disque local lors des expériences.
- `clean_subtitles_dict()` existe dans le code mais n'est pas actuellement invoqué par le pipeline principal.
- `--force` est le mécanisme actuel pour garantir la régénération des sorties lors des ajustements.

Boucle de développement locale suggérée :

```bash
python vad_lang_subtitle.py -t data/<votre_media>.mp4 --whisper-model small --force
```

Utilisez un modèle plus petit (`tiny`/`base`/`small`) pendant l'itération, puis passez à `large` pour la qualité finale.

---

## 🩺 Dépannage

| Symptôme | Action |
|---|---|
| `ffmpeg: command not found` | Installez FFmpeg et vérifiez avec `ffmpeg -version`. |
| Le premier lancement est très lent ou semble bloqué | Les téléchargements initiaux des modèles (Whisper + Silero) peuvent prendre du temps ; les relances sont plus rapides. |
| Erreurs CUDA / GPU | Essayez le mode CPU avec un modèle Whisper plus petit (`small`, `base`, `tiny`) et vérifiez une version PyTorch adaptée à votre environnement. |
| Les fichiers de sortie ne sont pas régénérés | Utilisez `--force` pour écraser les fichiers dérivés existants. |
| `pip install -r requirements.txt` échoue car le fichier est introuvable | Utilisez la commande d'installation manuelle indiquée dans Installation. |
| Étiquetage de langue imprécis sur des segments courts | Cela peut arriver sur des segments très courts / bruités ; la logique actuelle combine Whisper et Lingua mais reste sujette à des cas limites. |
| Sortie de sous-titres vide ou quasi vide | Vérifiez que l'entrée contient de la parole, inspectez le `.wav` extrait, puis relancez avec `--force` après validation de l'extraction FFmpeg. |
| Alternance de langue inattendue entre lignes voisines | Cela peut se produire sur des segments très courts ; envisagez un post-traitement par langue et durée minimale dans les outils en aval. |

---

## ⚠️ Limitations connues et hypothèses

- Le manifeste de dépendances n'est pas versionné (`requirements.txt`, `pyproject.toml` et `setup.py` sont absents de la racine au moment de la rédaction).
- La licence est déclarée comme MIT dans le README, mais aucun fichier `LICENSE` autonome n'est actuellement présent.
- Lingua est explicitement initialisé avec `EN/ZH/JA/AR` dans le flux principal, alors que les valeurs par défaut du helper incluent davantage de codes.
- Aucun test/benchmark automatisé n'est actuellement commité, la validation est principalement manuelle.
- Des scripts historiques sont présents à la racine et dans `archived/` ; seul `vad_lang_subtitle.py` doit être considéré comme actif, sauf si vous expérimentez volontairement.

---

## 🗺 Feuille de route

- Ajouter et maintenir un `requirements.txt` ou `pyproject.toml` épinglé.
- Ajouter des tests automatisés pour la logique de segmentation et de nettoyage des timestamps.
- Ajouter de la documentation de benchmark et d'évaluation qualité pour les cas limites multilingues.
- Ajouter un support optionnel de fichier de configuration au lieu d'un comportement uniquement basé sur des valeurs par défaut codées.
- Étendre l'ensemble des README i18n dans `i18n/` et garder la barre de langue synchronisée.
- Clarifier et unifier le comportement de sélection des langues entre la configuration du détecteur et les valeurs par défaut du helper.
- Ajouter un fichier `LICENSE` formel pour correspondre à la déclaration du README.

---

## 🔗 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) pour la transcription vocale
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) pour une détection d'activité vocale robuste
- [Lingua](https://github.com/pemistahl/lingua-java) pour une identification de langue très précise

---

## 🤝 Contribuer

1. Forkez et clonez
2. Créez une branche : `git checkout -b feat/your-idea`
3. Committez et poussez
4. Ouvrez une PR

Pour des changements importants, incluez :

- Une courte description du changement comportemental attendu
- Un exemple de commande reproductible
- Des extraits de sous-titres avant/après si pertinent

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- Ouvrez une issue pour les rapports de bugs, questions d'utilisation et demandes de fonctionnalités.
- Utilisez les options de soutien ci-dessus pour les demandes de sponsoring et de dons.

---

## 📄 Licence

MIT © Lachlan Chen
