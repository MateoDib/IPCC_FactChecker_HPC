import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import torch
import psutil
from pipeline import pipe  # Pipeline pour le LLM

# Template de prompt
question_prompt_template = """
Vous êtes un expert en analyse de texte scientifique et en climatologie. Votre tâche consiste à formuler une **question précise et pertinente** qui permettra de vérifier les informations ou affirmations contenues dans un extrait spécifique d'un article de presse, en vous basant uniquement sur les rapports du GIEC.

### Instructions :
1. Analysez attentivement le texte de l'extrait et identifiez toutes les affirmations clés ou les informations susceptibles d'être vérifiées dans les rapports du GIEC.
2. Formulez une **question claire, concise et spécifique** qui cible directement ces affirmations clés.
3. La question doit permettre de trouver les données ou les arguments nécessaires dans les rapports du GIEC pour valider ou invalider les affirmations de l'extrait.
4. Ne reformulez pas le texte de l'extrait sous forme de question générique. Concentrez-vous sur les points précis nécessitant une vérification scientifique.
5. Votre réponse doit être strictement au format suivant, sans ajouter de commentaire ou d'explication :
"Question : <votre_question>"

### Contexte d'analyse :
Extrait : "{current_phrase}"
Contexte élargi : "{context}"

Répondez strictement au format spécifié.
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

# Générer un prompt formaté
def generate_question_prompt(current_phrase, context):
    return question_prompt_template.format(current_phrase=current_phrase, context=context)

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

def prepare_prompts_for_files(input_paths):
    """
    Prépare les prompts pour plusieurs fichiers CSV.
    - input_paths peut être un chemin de répertoire ou une liste de fichiers.
    """
    article_prompts = {}

    # Identifier les fichiers
    if isinstance(input_paths, str):  # Si un répertoire est fourni
        file_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith(".csv")]
    elif isinstance(input_paths, list):  # Si une liste de fichiers est fournie
        file_paths = input_paths
    else:
        raise ValueError("input_paths doit être un chemin de répertoire ou une liste de fichiers.")

    # Préparer les prompts
    skipped_files = 0
    for file_path in tqdm(file_paths, desc="Préparation des fichiers"):
        df = pd.read_csv(file_path)

        # Vérifiez les colonnes nécessaires
        if not {"binary_response", "current_phrase", "context"}.issubset(df.columns):
            print(f"[WARNING] Colonnes manquantes dans le fichier : {file_path}")
            skipped_files += 1
            continue

        # Filtrer les lignes où binary_response == "1"
        df["binary_response"] = df["binary_response"].astype(str)
        df = df[df["binary_response"] == "1"]

        # Vérifier si le DataFrame est vide
        if df.empty:
            print(f"[INFO] Aucun extrait pertinent dans le fichier : {file_path}")
            skipped_files += 1
            continue

        # Générer les prompts et ajouter une colonne
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(lambda x: {"prompt": generate_question_prompt(x["current_phrase"], x["context"])}, batched=False)
        df["prompt"] = list(dataset["prompt"])

        # Ajouter le DataFrame préparé
        article_prompts[file_path] = df

    print(f"[INFO] Fichiers ignorés : {skipped_files}/{len(file_paths)}")
    return article_prompts



# Traiter un batch global de prompts
def process_global_batch(batch_prompts):
    """Traite un batch global de prompts avec le pipeline."""
    try:
        outputs = pipe(batch_prompts, max_new_tokens=500, temperature=0.7, return_full_text=False)
        return [output[0]["generated_text"] for output in outputs]
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement d'un batch : {e}")
        return [""] * len(batch_prompts)

# Traiter les prompts de plusieurs fichiers
def process_prompts_multi_files(article_prompts, max_batch_size):
    """Traite les prompts de plusieurs fichiers en un batch global."""
    all_results = {file_name: [] for file_name in article_prompts.keys()}
    batch_prompts = []
    batch_metadata = []

    for file_name, df in article_prompts.items():
        for _, row in df.iterrows():
            prompt = row["prompt"]
            if len(batch_prompts) < max_batch_size:
                batch_prompts.append(prompt)
                batch_metadata.append((file_name, row.to_dict()))
            else:
                # Traiter le batch actuel
                outputs = process_global_batch(batch_prompts)
                for meta, output in zip(batch_metadata, outputs):
                    file_name, row_data = meta
                    row_data["question"] = output
                    all_results[file_name].append(row_data)

                # Réinitialiser le batch
                batch_prompts = [prompt]
                batch_metadata = [(file_name, row.to_dict())]

    # Traiter les derniers prompts restants
    if batch_prompts:
        outputs = process_global_batch(batch_prompts)
        for meta, output in zip(batch_metadata, outputs):
            file_name, row_data = meta
            row_data["question"] = output
            all_results[file_name].append(row_data)

    return all_results

# Fonction principale pour traiter plusieurs fichiers
def process_all_files(input_paths, output_dir):
    """
    Traite plusieurs fichiers.
    - input_paths peut être un répertoire ou une liste de fichiers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Préparer les prompts
    article_prompts = prepare_prompts_for_files(input_paths)
    if not article_prompts:
        print("[ERROR] Aucun fichier valide à traiter.")
        return

    # Traiter les prompts
    max_batch_size = calculate_dynamic_batch_size()
    all_results = process_prompts_multi_files(article_prompts, max_batch_size)

    # Sauvegarder les résultats
    for file_path, rows in all_results.items():
        if not rows:
            print(f"[WARNING] Aucun résultat pour le fichier : {file_path}")
            continue
        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, os.path.basename(file_path).replace(".csv", "_with_questions.csv"))
        df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans : {output_path}")
 
