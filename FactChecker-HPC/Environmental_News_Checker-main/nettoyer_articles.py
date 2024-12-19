import os
import re
import nltk
from nltk import sent_tokenize
from tqdm import tqdm
import psutil  # Pour surveiller les ressources CPU
import torch  # Pour surveiller la mémoire GPU
import pipeline  # Pipeline contenant le modèle LLM

# Charger le pipeline LLM avec optimisation GPU
pipe = pipeline.pipe

# Dossier contenant les fichiers .txt à traiter
input_text_dir = "/home2020/home/beta/aebeling/Data/presse/articles_brutes"
output_text_dir = "/home2020/home/beta/aebeling/Data/presse/articles"
os.makedirs(output_text_dir, exist_ok=True)

# Template de prompt
rewrite_prompt_template = """
Vous êtes chargé de retranscrire exactement les phrases suivantes sans apporter aucune modification au contenu.
Votre seule tâche est d'améliorer uniquement la mise en forme, comme la suppression des espaces inutiles, des sauts de ligne excessifs,
ou d'autres artefacts visuels qui nuisent à la lisibilité.

**Bloc à retranscrire** :
{block}

**Texte retranscrit attendu** :
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

# Nettoyer les noms de fichiers
def clean_file_name(file_name):
    """Nettoie le nom du fichier en supprimant les underscores superflus."""
    file_name = re.sub(r"_+", " ", file_name)  # Remplacer plusieurs underscores consécutifs par un espace
    file_name = file_name.strip()  # Supprimer les espaces de début et fin
    file_name = re.sub(r"\s+", " ", file_name)  # Remplacer plusieurs espaces consécutifs par un seul
    return file_name.replace(" ", "_")  # Remplacer les espaces par un underscore pour un nom de fichier valide

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

# Reconstituer des blocs de phrases cohérentes
def build_coherent_blocks(text, block_size=5):
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i + block_size]) for i in range(0, len(sentences), block_size)]

# Supprimer le mot 'assistant' de la réponse
def clean_llm_output(response):
    return re.sub(r"^\s*assistant\s*", "", response, flags=re.IGNORECASE).strip()

# Traiter un batch de prompts avec le pipeline
def process_prompts_with_pipeline(prompts):
    """Traite un batch de prompts avec le pipeline."""
    try:
        outputs = pipe(prompts, max_new_tokens=500, temperature=0.7, return_full_text=False)
        return [clean_llm_output(output[0]["generated_text"]) for output in outputs]
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement d'un batch : {e}")
        return [""] * len(prompts)

# Traiter les prompts de plusieurs articles en batchs
def process_prompts_multi_articles(article_prompts, max_batch_size):
    """Traite les prompts de plusieurs fichiers en batch global."""
    all_results = {file_name: [] for file_name in article_prompts.keys()}
    batch_prompts = []
    batch_metadata = []

    total_prompts = sum(len(prompts) for prompts in article_prompts.values())
    with tqdm(total=total_prompts, desc="Traitement des prompts") as pbar:
        for file_name, prompts in article_prompts.items():
            for prompt in prompts:
                if len(batch_prompts) < max_batch_size:
                    batch_prompts.append(prompt)
                    batch_metadata.append(file_name)
                else:
                    # Traiter le batch actuel
                    outputs = process_prompts_with_pipeline(batch_prompts)
                    for meta, output in zip(batch_metadata, outputs):
                        all_results[meta].append(output)

                    # Réinitialiser le batch
                    batch_prompts = [prompt]
                    batch_metadata = [file_name]

                pbar.update(1)

        # Traiter les derniers prompts restants
        if batch_prompts:
            outputs = process_prompts_with_pipeline(batch_prompts)
            for meta, output in zip(batch_metadata, outputs):
                all_results[meta].append(output)
                pbar.update(len(batch_prompts))

    return all_results

# Fonction principale pour traiter tous les fichiers
def process_all_files_multi_articles(input_dir, output_dir):
    files_to_process = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
    article_prompts = {}

    # Construire les prompts pour chaque article
    print("[INFO] Construction des prompts...")
    for file_path in tqdm(files_to_process, desc="Préparation des fichiers"):
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        blocks = build_coherent_blocks(text_content, block_size=5)
        article_prompts[file_path] = [rewrite_prompt_template.format(block=block) for block in blocks]

    # Traiter tous les prompts
    max_batch_size = calculate_dynamic_batch_size()
    print(f"[INFO] Taille de batch maximale : {max_batch_size}")
    all_results = process_prompts_multi_articles(article_prompts, max_batch_size)

    # Sauvegarder les résultats avec des noms nettoyés
    for file_path, results in all_results.items():
        cleaned_name = clean_file_name(os.path.basename(file_path).replace(".txt", "_processed.txt"))
        output_path = os.path.join(output_dir, cleaned_name)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write("\n\n".join(results))
        print(f"[INFO] Saved results to {output_path}")
