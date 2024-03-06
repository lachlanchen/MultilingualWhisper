from lingua import Language, LanguageDetectorBuilder

# Initialize the Lingua language detector with specified languages
languages = [Language.ENGLISH, Language.CHINESE, Language.JAPANESE, Language.ARABIC]  # Adjust languages as needed
detector = LanguageDetectorBuilder.from_languages(*languages)\
    .with_minimum_relative_distance(0.9)\
    .build()

def detect_language_with_lingua(text):
    """
    Detects the language of a given text using Lingua.
    Returns the ISO 639-1 code of the detected language if detection is confident; otherwise, returns None.
    """
    try:
        language = detector.detect_language_of(text)
        return language.iso_code_639_1.name.lower()  # Use .name to get the ISO code as a string
    except Exception as e:
        print(f"Language detection failed: {e}")
        return None

if __name__ == "__main__":
    lang = detect_language_with_lingua("قش")
    print(lang)
