# Module 'embeddings_creation'
import json
from tqdm import tqdm

def generer_embeddings_rapport(chemin_rapport_json, embed_model):
    with open(chemin_rapport_json, 'r', encoding='utf-8') as file:
        data = json.load(file)

    embeddings = []
    sections = []
    titles = []
    
    # Générer les embeddings pour chaque section de texte
    for section in tqdm(data, desc="Generating embeddings for report sections"):
        text = section.get('text', 'vide')  # Récupérer le texte de la section
        title = section.get('title', 'Section sans titre')  # Titre par défaut si non défini

        if not text.strip():  # Vérifier si le texte est vide
            print(f"Avertissement : Section ignorée car elle est vide. Titre : '{title}'")
            continue

        # Générer l'embedding
        embedding = embed_model.encode(text, convert_to_tensor=True, device='cpu')
        section['embedding'] = embedding.cpu().numpy().tolist()  # Convertir en liste pour JSON
        
        embeddings.append(section['embedding'])
        sections.append(text)
        titles.append(title)
    
    # Sauvegarder les données avec les embeddings
    with open(chemin_rapport_json, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"Embeddings generated and saved to {chemin_rapport_json}")
    return embeddings, sections, titles



# Embedding function to efficiently process multiple texts
def embed_texts(texts, embed_model):
    embeddings = embed_model.encode(texts, convert_to_tensor=True, device='cpu')  # Ensure embedding is done on CPU
    return embeddings

