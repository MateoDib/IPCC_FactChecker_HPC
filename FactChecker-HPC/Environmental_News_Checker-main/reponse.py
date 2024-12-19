import os
import pandas as pd
from tqdm import tqdm
import torch
import psutil
from pipeline import pipe  # Pipeline HuggingFace préchargé

# Template de prompt pour les réponses
answer_prompt_template = """
Vous êtes un expert en climatologie et vous devez répondre à une question en vous basant uniquement sur les informations contenues dans les sections pertinentes d'un rapport du GIEC. Vous devez fournir une réponse claire, précise et strictement fondée sur les données disponibles dans ces sections.

**Instructions** :
1. Formulez une réponse concise et directement liée à la question posée, sans aucune introduction ou conclusion.
2. Justifiez votre réponse en citant précisément les sections pertinentes, avec les données textuelles ou chiffrées issues du rapport.
3. Ne répondez que dans le format strict spécifié ci-dessous.

### Format strict de réponse :
- **Résumé de la réponse** : (Une phrase concise répondant directement à la question.)
- **Justification** :
  - Section : [Titre ou Numéro de la section pertinente si disponible]
    - Détails : [Résumé des éléments pertinents extraits de cette section.]

### Question :
{question}

### Sections du rapport :
{consolidated_text}

Répondez maintenant en respectant strictement le format donné.
"""

# Logger les ressources CPU/GPU
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

# Mesurer la mémoire GPU utilisée par un batch
def measure_batch_memory_usage(batch_prompts):
    """Mesure la mémoire GPU réellement utilisée par un batch pour ajuster dynamiquement la taille."""
    if not torch.cuda.is_available():
        return 1  # Valeur par défaut si pas de GPU

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    try:
        pipe(batch_prompts, max_new_tokens=500, temperature=0.7, return_full_text=False)
    except Exception as e:
        print(f"[ERROR] Erreur lors du batch test : {e}")
        return 1

    final_memory = torch.cuda.memory_allocated()
    torch.cuda.empty_cache()

    batch_memory_used = final_memory - initial_memory
    average_memory_per_prompt = batch_memory_used / len(batch_prompts) / (1024**2)  # Convertir en MiB
    print(f"[INFO] Mémoire utilisée par prompt : {average_memory_per_prompt:.2f} MiB")
    return average_memory_per_prompt

# Calculer dynamiquement la taille optimale des batchs
def calculate_dynamic_batch_size():
    """Calcule dynamiquement la taille des batchs en fonction des ressources disponibles."""
    if not torch.cuda.is_available():
        return 4  # Taille par défaut pour CPU

    stats = torch.cuda.memory_stats()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = stats["allocated_bytes.all.current"]
    free_memory = total_memory - allocated_memory

    test_batch = ["Test prompt"] * 4
    memory_per_prompt = measure_batch_memory_usage(test_batch)

    if memory_per_prompt > 0:
        max_batch_size = int(free_memory // (memory_per_prompt * (1024**2)))
        print(max_batch_size)
    else:
        max_batch_size = 4  # Taille de batch par défaut

    return max(1, min(max_batch_size, 2000))

# Générer un prompt formaté
def generate_answer_prompt(question, consolidated_text):
    return answer_prompt_template.format(question=question, consolidated_text=consolidated_text)

# Préparer les prompts pour plusieurs fichiers
def prepare_prompts_for_files(input_paths):
    """Préparer les prompts pour les fichiers spécifiés."""
    article_prompts = {}

    # Identifier les fichiers
    if isinstance(input_paths, str):  # Si un répertoire est fourni
        file_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith("_resume_sections_results.csv")]
    elif isinstance(input_paths, list):  # Si une liste de fichiers est fournie
        file_paths = input_paths
    else:
        raise ValueError("input_paths doit être un chemin de répertoire ou une liste de fichiers.")

    for file_path in tqdm(file_paths, desc="Préparation des prompts"):
        df_questions = pd.read_csv(file_path)

        # Vérifier les colonnes nécessaires
        if not {"question", "sections_resumees"}.issubset(df_questions.columns):
            print(f"[WARNING] Colonnes manquantes dans le fichier : {file_path}")
            continue

        article_prompts[file_path] = [
            {
                **row,  # Conserver toutes les colonnes originales
                "prompt": generate_answer_prompt(row["question"], row["sections_resumees"])
            }
            for _, row in df_questions.iterrows()
        ]
    return article_prompts

# Traiter un batch global de prompts avec le pipeline
def process_global_batch(batch_prompts):
    """Traite un batch global de prompts avec le pipeline."""
    try:
        outputs = pipe([prompt["prompt"] for prompt in batch_prompts], max_new_tokens=500, temperature=0.7, return_full_text=False)
        for i, output in enumerate(outputs):
            batch_prompts[i]["response"] = output[0]["generated_text"]
        return batch_prompts
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement d'un batch : {e}")
        return [{"response": ""} for _ in batch_prompts]

# Traiter les prompts de plusieurs fichiers en batchs globaux
def process_prompts_multi_files(article_prompts, max_batch_size):
    """Traite les prompts de plusieurs fichiers en un batch global."""
    all_results = {file_name: [] for file_name in article_prompts.keys()}
    batch_prompts = []

    for file_name, prompts in article_prompts.items():
        for prompt in prompts:
            if len(batch_prompts) < max_batch_size:
                batch_prompts.append(prompt)
            else:
                # Traiter le batch actuel
                batch_results = process_global_batch(batch_prompts)
                all_results[file_name].extend(batch_results)

                # Réinitialiser le batch
                batch_prompts = [prompt]

    # Traiter les derniers prompts restants
    if batch_prompts:
        batch_results = process_global_batch(batch_prompts)
        for file_name, prompts in article_prompts.items():
            all_results[file_name].extend(batch_results)

    return all_results

# Fonction principale pour traiter plusieurs fichiers
def process_all_responses(input_paths, output_dir):
    """
    Traite plusieurs fichiers pour générer des réponses.
    - input_paths peut être un chemin de répertoire ou une liste de fichiers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Préparer les prompts pour tous les fichiers
    article_prompts = prepare_prompts_for_files(input_paths)

    # Traiter les prompts
    max_batch_size = calculate_dynamic_batch_size()
    all_results = process_prompts_multi_files(article_prompts, max_batch_size)

    # Sauvegarder les résultats
    for file_path, rows in all_results.items():
        if not rows:
            print(f"[WARNING] Aucun résultat pour le fichier : {file_path}")
            continue
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace("_resume_sections_results.csv", "_rag_results.csv"))
        df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans : {output_path}")
