"""
Multilingual GPQA Dataset Translation and Hub Upload Pipeline.

This script automates the process of fetching the English GPQA Diamond dataset,
translating its core columns into specified target languages using Google Translate,
and pushing the combined multilingual dataset to the Hugging Face Hub.
"""

import time
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from googletrans import Translator
from tqdm import tqdm
from huggingface_hub import login

HF_USERNAME = "YOUR_HUGGINGFACE_USERNAME"
DATASET_NAME = "Multilingual_gpqa"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

print("AUTHENTICATION")
hf_token = input("Paste your Hugging Face Token (Needs Write permissions) and press Enter: ")
login(token=hf_token)

print("\nLOADING DATA")
print("Loading GPQA Diamond Subset")

gpqa_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
df_english = gpqa_dataset['train'].to_pandas()

cols_to_translate = [
    'Question',
    'Correct Answer',
    'Incorrect Answer 1',
    'Incorrect Answer 2',
    'Incorrect Answer 3'
]
df_english = df_english[cols_to_translate]

target_languages = {
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'ru': 'Russian',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'th': 'Thai',
    'sw': 'Swahili',
    'bn': 'Bengali',
    'te': 'Telugu'
}

translator = Translator()

def translate_text(text, target_lang, retries=3):
    """
    Translates a given text string to the specified target language.

    Utilizes the Google Translate API with a built-in backoff mechanism
    to handle transient connection errors and prevent rate-limiting blocks.

    Args:
        text (str): The input string to be translated.
        target_lang (str): The destination language code (e.g., 'ru', 'zh-cn').
        retries (int, optional): The maximum number of attempts. Defaults to 3.

    Returns:
        str: The translated text string. If all retries fail or the input
             is invalid, the original un-translated text is returned.
    """
    if not isinstance(text, str) or not text.strip():
        return text

    for attempt in range(retries):
        try:
            result = translator.translate(text, dest=target_lang).text
            time.sleep(0.5)
            return result
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed to translate: {text[:30]}... Error: {e}")
                return text
            time.sleep(2)

print("\nTRANSLATION PHASE")
for lang_code, lang_name in target_languages.items():
    print(f"\nTranslating to {lang_name} ({lang_code})...")
    df_translated = df_english.copy()

    for col in cols_to_translate:
        print(f"Translating column: {col}")
        tqdm.pandas(desc=f"{col}")
        df_translated[col] = df_translated[col].progress_apply(lambda x: translate_text(x, lang_code))

    output_filename = f"gpqa_diamond_mt_{lang_code}.csv"
    df_translated.to_csv(output_filename, index=False)
    print(f"Saved {lang_name} translation to {output_filename}")


print("\nUPLOAD PHASE")
dataset_dict = {}
lang_codes = list(target_languages.keys())

for lang in lang_codes:
    try:
        df = pd.read_csv(f"gpqa_diamond_mt_{lang}.csv")
        split_name = lang.replace("-", "_")
        dataset_dict[split_name] = Dataset.from_pandas(df)
        print(f"Loaded {lang} successfully as split '{split_name}'.")
    except Exception as e:
        print(f"Skipping {lang}: {e}")

print("Adding original English to the split 'en'...")
dataset_dict["en"] = Dataset.from_pandas(df_english)

print(f"\nPushing combined dataset to {REPO_ID}...")
hf_dataset = DatasetDict(dataset_dict)
hf_dataset.push_to_hub(REPO_ID, private=False)

print(f"\nSuccess! Dataset uploaded to https://huggingface.co/datasets/{REPO_ID}")
