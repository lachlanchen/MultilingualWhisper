[English](../README.md) · [العربية](README.ar.md) · [Español](README.es.md) · [Français](README.fr.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Tiếng Việt](README.vi.md) · [中文 (简体)](README.zh-Hans.md) · [中文（繁體）](README.zh-Hant.md) · [Deutsch](README.de.md) · [Русский](README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# MultilingualWhisper

Un générateur de sous-titres prêt à l’emploi, construit sur OpenAI Whisper, étendu avec une détection de langue précise par segment et un raffinement adapté aux vidéos contenant des langues mixtes.

> Générez des sous-titres multilingues plus propres à partir de médias réels en langues mélangées grâce à une segmentation consciente de la langue.

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

> 🌍 **Documentation multilingue disponible** : anglais + 10 variantes de README traduites dans [`i18n/`](i18n/), liées dans la barre de langue ci-dessus.

### Langues de la documentation

| Locale | Fichier |
| --- | --- |

| Focus | Value |
| --- | --- |
| Input | Audio/vidéo compatibles FFmpeg |
| Pipeline | Segmentation VAD -> transcription Whisper -> raffinement Lingua |
| Output | `*.wav`, `*.srt` et `*.json` normalisés |
| Best use | Sous-titres multilingues avec tags de langue par segment |

---

## Table des matières

- [Vue d’ensemble](#-vue-densemble)
- [En bref](#en-bref)
- [Fonctionnalités clés](#-fonctionnalités-clés)
- [Flux du pipeline](#-flux-du-pipeline)
- [Structure du projet](#-structure-du-projet)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Démarrage rapide](#-démarrage-rapide)
- [Guide de sélection des modèles](#-guide-de-sélection-des-modèles)
- [Utilisation](#-utilisation)
- [Configuration](#-configuration)
- [Format de sortie](#-format-de-sortie)
- [Exemples](#-exemples)
- [Notes de développement](#-notes-de-développement)
- [Dépannage](#-dépannage)
- [Limites connues et hypothèses](#-limites-connues-et-hypothèses)
- [Feuille de route](#-feuille-de-route)
- [Remerciements](#-remerciements)
- [Contribution](#-contribution)
- [Support](#-support)
- [Contact](#-contact)
- [Licence](#-licence)

---

## ✨ Vue d’ensemble

`MultilingualWhisper` est un pipeline CLI Python centré sur [`vad_lang_subtitle.py`](vad_lang_subtitle.py). Il combine :

- Silero VAD pour la segmentation de la parole
- OpenAI Whisper pour la transcription et la prédiction initiale de langue
- Lingua pour l’affinage de la langue basé sur le texte
- FFmpeg pour l’extraction, la normalisation et le traitement des médias

Les sorties principales sont des fichiers de sous-titres en `.srt` et `.json`, ainsi qu’un audio `.wav` extrait et normalisé.

### En bref

| Élément | Détails |
|---|---|
| Point d’entrée principal | `vad_lang_subtitle.py` |
| Entrée | Vidéo/audio pris en charge par FFmpeg |
| Sortie | `*.wav`, `*.srt`, `*.json` |
| Flux principal | VAD -> Whisper -> Lingua -> raffinement |
| Cas d’usage typique | Génération de sous-titres en langues mixtes |

---

## 🚀 Fonctionnalités clés

- **Pipeline Silero VAD -> Whisper**  
  La détection d’activité vocale (VAD) découpe l’audio en segments de parole, puis Whisper transcrit chaque segment.

- **Détection de langue fine**  
  Utilise [Lingua](https://github.com/pemistahl/lingua-java) avec le détecteur natif de Whisper pour étiqueter chaque segment (et même des mots individuels) avec des codes ISO (`en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr`, ...).

- **Raffinement intelligent des segments**  
  Le nettoyage des timestamps garantit l’absence de trous ou de chevauchements. La découpe par ponctuation casse les transcriptions longues aux virgules, points, points d’interrogation, etc. Les fusions VAD réalignent les mots sur les blocs VAD pour des sous-titres plus fluides. La segmentation sensible à la longueur applique des limites spécifiques à la langue.

- **Sous-titres multilingues**  
  Produit à la fois `.srt` et `.json`, en conservant les tags de langue par segment afin de pouvoir styliser ou filtrer par langue dans les lecteurs/éditeurs en aval.

- **Gestion robuste des médias**  
  Extrait et normalise automatiquement l’audio via FFmpeg, tente de réparer les conteneurs cassés, et applique une normalisation dynamique (`dynaudnorm`) pour des transcriptions plus nettes.

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

1. Analyse les arguments CLI (`--video-path`, `--whisper-model`, `--force`).
2. Résout les chemins de sortie à partir du nom de base de l’entrée.
3. Extrait/normalise l’audio via FFmpeg.
4. Charge Silero VAD (`torch.hub`) et le modèle Whisper.
5. Effectue une première passe de transcription sur les chunks VAD.
6. Fusionne/raffine les segments, puis effectue une seconde passe de transcription sur les plages fusionnées.
7. Applique la réduction de longueur des sous-titres et le nettoyage des timestamps.
8. Enregistre `.srt` et `.json`.

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

> ⚠️ Remarque : un précédent README mentionnait `requirements.txt`, mais ce fichier est actuellement absent à la racine du dépôt.

---

## ✅ Prérequis

- Python `3.10+` (testé avec des environnements 3.x modernes)
- `ffmpeg` installé et disponible dans le `PATH`
- CPU/GPU + RAM suffisants pour le modèle Whisper sélectionné (pour `large`, un GPU est fortement recommandé)
- Accès Internet au premier lancement pour récupérer les poids du modèle Whisper et les assets Silero VAD (`torch.hub`)

Les packages Python utilisés par le script incluent :

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

1. **Clonez ce dépôt**

```bash
git clone git@github.com:lachlanchen/whisper_with_lang_detect.git
cd whisper_with_lang_detect
```

2. **Créez et activez un environnement virtuel**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Installez les dépendances**

```bash
pip install -r requirements.txt
```

Si `requirements.txt` est toujours absent dans votre checkout, installez manuellement les dépendances runtime principales :

```bash
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
```

Et assurez-vous que FFmpeg est installé au niveau système.

---

## ⚡ Démarrage rapide

Si vous voulez le chemin le plus court entre le clone et les sous-titres :

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio openai-whisper lingua-language-detector tqdm
python vad_lang_subtitle.py -t path/to/video.mp4 --whisper-model small --force
```

Conseil : utilisez `small` pendant l’itération, puis passez à `large` pour la qualité finale.

Artifacts attendus à côté de votre média d’entrée :

- `*.wav` audio extrait et normalisé
- `*.srt` fichier de sous-titres pour lecteurs/éditeurs
- `*.json` métadonnées structurées de sous-titres multilingues

---

## 🎚 Guide de sélection des modèles

Choisissez un modèle Whisper selon vos objectifs vitesse vs qualité :

| Modèle | Vitesse | Qualité | Usage recommandé |
|---|---|---|---|
| `tiny` / `base` | La plus rapide | La plus faible | Smoke tests rapides et validation du pipeline |
| `small` | Rapide | Bonne | Itération quotidienne et développement local |
| `medium` | Moyenne | Meilleure | Workflows de production équilibrés |
| `large` (par défaut) | La plus lente | La meilleure | Exports finaux de sous-titres pour la qualité maximale |

Schéma pratique :

1. Itérer avec `small --force`
2. Valider les timings et les tags de langue
3. Relancer avec `large --force` pour la sortie de livraison

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
| `--force` | — | Non | Relancer même si `.wav`, `.srt` ou `.json` existent déjà |

### Comportement du traitement

- Les noms de sortie sont dérivés du chemin de base de l’entrée.
- Pour `input.mp4`, les sorties sont `input.wav` (audio normalisé), `input.srt` (sous-titres horodatés) et `input.json` (métadonnées incluant `start`, `end`, `lang`, `text`, éventuellement les timings mot à mot).
- Si `.srt` ou `.json` existent déjà, le traitement est ignoré sauf si `--force` est défini.

---

## ⚙️ Configuration

La configuration actuelle est principalement pilotée par la CLI et les valeurs par défaut du code :

| Zone de config | Comportement actuel |
|---|---|
| Modèle Whisper | `--whisper-model` (par défaut `large`) |
| Taux d’échantillonnage de traitement | Codé en dur à `16000` pour VAD/transcription |
| Extraction FFmpeg | WAV mono, `44100 Hz`, avec `dynaudnorm=f=100` |
| Détecteur Lingua | Initialisé pour `ENGLISH`, `CHINESE`, `JAPANESE`, `ARABIC` dans le flux principal |
| Valeurs par défaut de filtrage côté Whisper | Inclut `en`, `zh`, `ja`, `ar`, `yue`, `ko`, `vi`, `es`, `fr` |

Remarque d’hypothèse : les listes de langues dans les valeurs par défaut des helpers et la configuration principale du détecteur ne sont pas totalement identiques ; ce README préserve le comportement actuel tel qu’implémenté.

Détail d’implémentation supplémentaire du script actuel :

- `torch.set_num_threads(1)` est appliqué à l’exécution.
- Le modèle VAD est chargé depuis `snakers4/silero-vad` via `torch.hub.load(...)`.
- Le nettoyage des segments supprime les entrées où la langue est `und` ou le texte est vide.

---

## 📦 Format de sortie

L’outil écrit deux artifacts de sous-titres par média d’entrée :

- `*.srt` : texte de sous-titres standard avec timestamps `HH:MM:SS,mmm`.
- `*.json` : liste structurée de sous-titres contenant des timestamps formatés et des tags de langue.

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

Remarques :

- `start`/`end` sont sérialisés en chaînes de style SRT dans la sortie JSON.
- `words` peut être présent selon l’étape de traitement/raffinement du segment.
- Une valeur `lang` égale à `und` peut apparaître pour des portions dont la langue est incertaine.

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
- Les fichiers historiques (`*.old`, `*.shorterlength*`, `archived/`) sont utiles comme référence, mais paraissent non canoniques.
- Il n’y a actuellement ni scaffolding de packaging projet (`pyproject.toml`, `setup.py`) ni suite CI/tests commitée.
- `data/` contient de gros artifacts média d’exemple ; gardez en tête la taille du dépôt et l’usage disque local pendant les expérimentations.
- `clean_subtitles_dict()` existe dans le code mais n’est actuellement pas appelé par le pipeline principal.
- `--force` est le mécanisme actuel pour garantir la régénération des sorties lors d’un ajustement itératif.

Boucle de dev locale suggérée :

```bash
python vad_lang_subtitle.py -t data/<your_media>.mp4 --whisper-model small --force
```

Utilisez un modèle plus petit (`tiny`/`base`/`small`) pendant l’itération, puis passez à `large` pour la qualité de sortie finale.

---

## 🩺 Dépannage

| Symptôme | Que faire |
|---|---|
| `ffmpeg: command not found` | Installez FFmpeg et vérifiez avec `ffmpeg -version`. |
| Le premier lancement est très lent ou semble bloqué | Les téléchargements initiaux des modèles (Whisper + Silero) peuvent prendre du temps ; les relances sont plus rapides. |
| Erreurs CUDA / GPU | Essayez un repli CPU en utilisant un modèle Whisper plus petit (`small`, `base`, `tiny`) et assurez-vous d’avoir une build PyTorch compatible avec votre environnement. |
| Les fichiers de sortie ne sont pas régénérés | Utilisez `--force` pour écraser les fichiers dérivés existants. |
| `pip install -r requirements.txt` échoue car le fichier est introuvable | Utilisez la commande d’installation manuelle des dépendances montrée dans Installation. |
| Étiquetage de langue imprécis sur des segments courts | Cela peut arriver sur des portions extrêmement courtes/bruitées ; la logique actuelle combine Whisper et Lingua mais a encore des cas limites. |
| Sortie de sous-titres vide ou quasi vide | Vérifiez que l’entrée contient de la parole, inspectez le `.wav` extrait, puis réessayez avec `--force` après validation de l’extraction FFmpeg. |
| Changement inattendu de langue entre des lignes voisines | Cela peut se produire sur des segments très courts ; envisagez une post-fusion dans les outils en aval selon la langue et une durée minimale. |
| L’extraction FFmpeg échoue sur un média endommagé | Le script réessaie après réparation du conteneur (`-c copy -movflags +faststart`), mais des fichiers très corrompus peuvent quand même échouer. |

Diagnostics rapides :

```bash
python --version
ffmpeg -version
python -c "import torch, whisper, torchaudio, tqdm; print('python deps ok')"
```

---

## ⚠️ Limites connues et hypothèses

- Le manifeste de dépendances n’est pas commité (`requirements.txt`, `pyproject.toml` et `setup.py` sont absents à la racine du dépôt au moment de la rédaction).
- La licence est déclarée MIT dans le README, mais un fichier `LICENSE` autonome n’est actuellement pas présent.
- Lingua est explicitement initialisé avec `EN/ZH/JA/AR` dans le flux principal, tandis que les valeurs par défaut des helpers incluent davantage de codes candidats.
- Aucun test/benchmark automatisé n’est actuellement commité, la validation est donc principalement manuelle.
- Des scripts historiques sont présents à la racine et dans `archived/` ; seul `vad_lang_subtitle.py` doit être considéré comme actif sauf expérimentation volontaire.
- Le script affiche actuellement des logs d’exécution verbeux et un debug par segment ; c’est le comportement attendu dans l’implémentation actuelle.

---

## 🗺 Feuille de route

- Ajouter et maintenir un `requirements.txt` ou `pyproject.toml` épinglé.
- Ajouter des tests automatisés pour la segmentation et la logique de nettoyage des timestamps.
- Ajouter une documentation de benchmark et d’évaluation qualité pour les cas limites multilingues.
- Ajouter une prise en charge optionnelle d’un fichier de configuration au lieu d’un comportement uniquement basé sur les valeurs par défaut du code.
- Étendre l’ensemble des README i18n dans `i18n/` et garder les barres de langue synchronisées.
- Clarifier et unifier le comportement de sélection des langues entre la configuration du détecteur et les valeurs par défaut des helpers.
- Ajouter un fichier `LICENSE` formel pour correspondre à la déclaration du README.

---

## 🔗 Remerciements

- [OpenAI Whisper](https://github.com/openai/whisper) pour le speech-to-text
- [Snakers4/Silero-VAD](https://github.com/snakers4/silero-models) pour une détection d’activité vocale robuste
- [Lingua](https://github.com/pemistahl/lingua-java) pour l’identification de langue haute précision

---

## 🤝 Contribution

1. Forkez et clonez
2. Créez une branche : `git checkout -b feat/your-idea`
3. Committez et poussez
4. Ouvrez une PR

Pour les changements substantiels, incluez :

- Une courte description du changement de comportement attendu
- Un exemple de commande reproductible
- Des extraits de sous-titres avant/après quand c’est pertinent

---

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📫 Contact

- Ouvrez une issue pour les rapports de bug, les questions d’usage et les demandes de fonctionnalités.
- Utilisez les options de support ci-dessus pour les demandes de sponsoring et de don.

---

## 📄 Licence

MIT © Lachlan Chen
