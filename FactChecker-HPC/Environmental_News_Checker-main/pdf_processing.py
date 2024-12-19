"""
Ce script fournit un pipeline de traitement pour extraire du texte à partir d'un PDF, le nettoyer et le découper en sections.
Il enregistre ensuite les sections dans un fichier JSON.

Fonctionnalités principales :
- Extraction de texte à partir de fichiers PDF
- Nettoyage du texte extrait pour supprimer les espaces et numéros de page
- Découpage du texte en sections
- Sauvegarde des sections dans un fichier JSON

"""
import json
from pdfminer.high_level import extract_text
import re
import nltk
import os

# SpÈcifiez explicitement le chemin des donnÈes NLTK
nltk_data_path = "/home2020/home/beta/aebeling/nltk_data"
nltk.data.path.append(nltk_data_path)

# VÈrifiez que le chemin existe et que les donnÈes sont accessibles
if not os.path.exists(nltk_data_path):
    raise FileNotFoundError(f"Le repertoire specifie pour les donnees NLTK ({nltk_data_path}) est introuvable.")

# Fonction pour nettoyer le texte de manière simple
def clean_text(text: str) -> str:
    """
    Nettoie un texte en supprimant les numéros de page, les espaces multiples et les sauts de ligne supplémentaires.

    Args:
        text (str): Texte à nettoyer.

    Returns:
        str: Texte nettoyé.
    """
    text = re.sub(r'\bPage\s+\d+\b', '', text)  # Supprimer les numéros de page
    text = re.sub(r'\s+', ' ', text)  # Supprimer les espaces multiples
    text = re.sub(r'\n+', '\n', text)  # Supprimer les sauts de ligne en excès
    return text


# Function to split text into optimized chunks using NLTK for sentence tokenization
def split_text_into_chunks(text: str, max_tokens: int = 382) -> list[dict]:
    """
    Splits a text into optimized chunks for embedding generation, ensuring each chunk is within the token limit.

    Args:
        text (str): The input text to split into chunks.
        max_tokens (int): The maximum number of tokens for each chunk (default is 382 for the model).

    Returns:
        list: A list of dictionaries containing the index and content of each chunk.
    """
    # Split the text into sentences using NLTK
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    chunk_index = 0

    for sentence in sentences:
        # Check if adding the next sentence exceeds the max token limit
        if len(current_chunk) + len(sentence) <= max_tokens:
            current_chunk += sentence + " "
        else:
            # Add the current chunk to the list
            chunks.append({
                "title": f"Chunk {chunk_index}",
                "text": current_chunk.strip()
            })
            chunk_index += 1
            # Start a new chunk with the current sentence
            current_chunk = sentence + " "

    # Add the last chunk if it has any remaining text
    if current_chunk:
        chunks.append({
            "title": f"Chunk {chunk_index}",
            "text": current_chunk.strip()
        })

    return chunks

# Pipeline principal de traitement
def process_pdf_to_index(chemin_rapport_pdf: str, chemin_output_json: str) -> None:
    """
    Traite un fichier PDF pour en extraire du texte, le nettoyer, le découper en sections, 
    puis sauvegarder les sections dans un fichier JSON.

    Args:
        chemin_rapport_pdf (str): Chemin vers le fichier PDF à traiter.
        chemin_output_json (str): Chemin du fichier JSON o√π sauvegarder les sections extraites.
    """
    # Extraction du texte du PDF
    raw_text = extract_text(chemin_rapport_pdf)

    # Nettoyage du texte extrait
    cleaned_text = clean_text(raw_text)

    # Découpage du texte en sections
    sections = split_text_into_chunks(cleaned_text)

    # Sauvegarde des sections dans le fichier de sortie JSON
    with open(chemin_output_json, 'w', encoding='utf-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)
    print(f"Sections sauvegardées dans {chemin_output_json}")
