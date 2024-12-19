import os
import pandas as pd
from tqdm import tqdm
import torch
import psutil
from nltk import sent_tokenize
from pipeline import pipe  # Pipeline pour le LLM
from generate_context_windows import generate_context_windows
from llms import parsed_responses  # Fonction pour parser les réponses du LLM

# Template de prompt
prompt_template = """
Vous êtes un expert en traitement de texte chargé d'analyser le contenu suivant. Votre tâche est de répondre strictement au format JSON spécifié, sans aucune introduction ni commentaire supplémentaire.

### Tâches :
1. Si le texte mentionne directement ou indirectement des sujets liés à l'environnement, au changement climatique, au réchauffement climatique, ou à des organisations, événements ou accords associés (par exemple, le GIEC, les conférences COP, les accords de Paris, etc.), répondez avec une valeur '1'. Sinon, répondez avec '0'.
2. Identifiez tous les sujets abordés dans le texte et listez-les.

### Format strict de réponse :
{{
    "Réponse": <0 ou 1>,
    "Liste des sujets abordés dans la phrase": [
        "sujet_1",
        "sujet_2",
        "sujet_3"
    ]
}}

### Texte à analyser :
Phrase : "{current_phrase}"
Contexte : "{context}"

Répondez maintenant en respectant strictement le format spécifié.
"""

# Logger l'état des ressources CPU/GPU
def log_resources(stage):
    print(f"\n[LOG - {stage}]")
    cpu_mem = psutil.virtual_memory()
    print(f"CPU Utilisation : {psutil.cpu_percent()}% | RAM utilisée : {cpu_mem.used / 1e9:.2f} GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            stats = torch.cuda.memory_stats(f"cuda:{i}")
            total_mem = torch.cuda.get_device_properties(f"cuda:{i}").total_memory / 1e9
            allocated_mem = stats["allocated_bytes.all.current"] / 1e9
            print(f"GPU {i} - Utilisation : {allocated_mem:.2f} GB / {total_mem:.2f} GB")

# Calculer la taille dynamique des batchs
def calculate_dynamic_batch_size():
    """Calcule dynamiquement la taille des batchs en fonction des ressources GPU disponibles."""
    if not torch.cuda.is_available():
        return 4  # Taille par défaut si pas de GPU

    stats = torch.cuda.memory_stats()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = stats["allocated_bytes.all.current"]
    free_memory = total_memory - allocated_memory

    # Estimation mémoire par prompt (100 MiB par prompt par défaut)
    max_batch_size = max(1, min(int(free_memory // (100 * 1024**2)), 2000))
    print(f"[INFO] Batch size dynamique calculé : {max_batch_size}")
    return max_batch_size

# Générer un prompt formaté
def generate_prompt(current_phrase, context):
    return prompt_template.format(current_phrase=current_phrase, context=context)

# Construire les prompts pour plusieurs fichiers
def prepare_prompts_for_files(file_paths):
    article_prompts = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = sent_tokenize(text)
        splitted_text = generate_context_windows(sentences)
        prompts = [
            {"id": idx, "current_phrase": entry["current_phrase"], "context": entry["context"], "prompt": generate_prompt(entry["current_phrase"], entry["context"])}
            for idx, entry in enumerate(splitted_text)
        ]
        article_prompts[file_path] = prompts
    return article_prompts

# Traiter un batch global de prompts
def process_global_batch(batch_prompts):
    try:
        outputs = pipe(batch_prompts, max_new_tokens=500, temperature=0.7, return_full_text=False)
        return [output[0]["generated_text"] for output in outputs]
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement du batch : {e}")
        return [""] * len(batch_prompts)

# Traiter les prompts de plusieurs fichiers avec un batch global
def process_prompts_multi_articles(article_prompts, max_batch_size):
    all_results = {file_name: [] for file_name in article_prompts.keys()}
    batch_prompts = []
    batch_metadata = []

    for file_name, prompts in article_prompts.items():
        for prompt_data in prompts:
            if len(batch_prompts) < max_batch_size:
                batch_prompts.append(prompt_data["prompt"])
                batch_metadata.append((file_name, prompt_data))
            else:
                # Traiter le batch actuel
                outputs = process_global_batch(batch_prompts)
                for meta, output in zip(batch_metadata, outputs):
                    file_name, data = meta
                    data["climate_related"] = output
                    all_results[file_name].append(data)

                # Réinitialiser le batch
                batch_prompts = [prompt_data["prompt"]]
                batch_metadata = [(file_name, prompt_data)]

    # Traiter les derniers prompts restants
    if batch_prompts:
        outputs = process_global_batch(batch_prompts)
        for meta, output in zip(batch_metadata, outputs):
            file_name, data = meta
            data["climate_related"] = output
            all_results[file_name].append(data)

    return all_results

# Fonction principale pour traiter plusieurs fichiers
def process_all_files(input_paths, output_dir):
    """
    Traite plusieurs fichiers.
    - input_paths peut être un répertoire ou une liste de fichiers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Vérifiez si input_paths est un répertoire ou une liste
    if isinstance(input_paths, str):  # Cas d'un chemin de répertoire
        file_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith(".txt")]
    elif isinstance(input_paths, list):  # Cas d'une liste de fichiers
        file_paths = input_paths
    else:
        raise ValueError("input_paths doit être un chemin de répertoire ou une liste de fichiers.")

    # Préparer les prompts
    article_prompts = prepare_prompts_for_files(file_paths)

    # Traiter les prompts
    max_batch_size = calculate_dynamic_batch_size()
    all_results = process_prompts_multi_articles(article_prompts, max_batch_size)

    # Sauvegarder les résultats avec le parsing
    for file_path, results in all_results.items():
        df = pd.DataFrame(results)
        parsed_df = parsed_responses(df)  # Appel au module pour parser les réponses
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace(".txt", "_analysis_results.csv"))
        parsed_df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans : {output_path}")
