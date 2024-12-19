import os
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import torch
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown
from sentence_transformers import SentenceTransformer, util
from embeddings_creation import embed_texts, generer_embeddings_rapport
from pipeline import pipe  # Pipeline partagé horizontalement pour le LLM

# Template de prompt
resume_prompt_template = """
Vous êtes un expert en climatologie. Votre tâche est de résumer précisément les informations importantes des sections fournies du rapport du GIEC, afin de répondre à une question spécifique. Le résumé doit être strictement basé sur les données textuelles et chiffrées contenues dans les sections.

**Instructions** :
1. Fournissez un résumé structuré, précis et concis des faits pertinents tirés des sections.
2. Limitez votre réponse aux informations fournies dans les sections. N'ajoutez aucune interprétation ou opinion.
3. Utilisez un format structuré et facile à analyser.

### Question :
"{question}"

### Sections du rapport :
{retrieved_sections}

Répondez strictement en suivant le format donné.
"""

# Initialiser NVML
try:
    nvmlInit()
except Exception as e:
    print(f"[ERROR] Impossible d'initialiser NVML : {e}")

# Logger les ressources CPU/GPU
def log_resources(stage):
    """
    Log l'état des ressources CPU et GPU.
    """
    print(f"\n[LOG - {stage}]")
    cpu_mem = psutil.virtual_memory()
    print(f"CPU Utilisation : {psutil.cpu_percent()}% | RAM utilisée : {cpu_mem.used / 1e9:.2f} GB")

    try:
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU {i} - Mémoire utilisée : {memory_info.used / 1024**2:.2f} MiB / {memory_info.total / 1024**2:.2f} MiB | Utilisation GPU : {utilization.gpu}%")
    except Exception as e:
        print(f"[ERROR] Impossible de récupérer les données GPU : {e}")

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
def generate_resume_prompt(question, retrieved_sections):
    return resume_prompt_template.format(question=question, retrieved_sections=retrieved_sections)

# Préparer les prompts pour plusieurs fichiers
def prepare_prompts_for_files(input_paths, embed_model, embeddings_rapport, sections_rapport):
    article_prompts = {}

    if isinstance(input_paths, str):  # Si un répertoire est fourni
        file_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith("_with_questions.csv")]
    elif isinstance(input_paths, list):  # Si une liste de fichiers est fournie
        file_paths = input_paths
    else:
        raise ValueError("input_paths doit être un chemin de répertoire ou une liste de fichiers.")

    for file_path in tqdm(file_paths, desc="Préparation des prompts"):
        df_questions = pd.read_csv(file_path)
        log_resources("Début du filtrage des sections pertinentes")
        dataset = filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport)
        article_prompts[file_path] = [
            {
                **row,
                "retrieved_sections": row["retrieved_sections"]
            }
            for row in dataset
        ]
    return article_prompts

# Filtrer les sections pertinentes
def filtrer_sections_pertinentes(df_questions, embed_model, embeddings_rapport, sections_rapport, top_k=5):
    def process_question(row):
        question_embedding = embed_texts([row['question']], embed_model)[0]
        similarites = util.cos_sim(question_embedding, torch.tensor(embeddings_rapport, device='cpu'))
        top_k_indices = torch.topk(similarites, k=top_k, dim=-1).indices[0].tolist()
        return " ".join([sections_rapport[i] for i in top_k_indices])

    df_questions["retrieved_sections"] = df_questions.apply(process_question, axis=1)
    return df_questions.to_dict("records")

# Traiter un batch global de prompts
def process_global_batch(batch_prompts):
    try:
        log_resources("Début du traitement des batchs")
        outputs = pipe(batch_prompts, max_new_tokens=500, temperature=0.7, return_full_text=False)
        log_resources("Fin du traitement des batchs")
        return [output[0]["generated_text"] for output in outputs]
    except Exception as e:
        print(f"[ERROR] Erreur lors du traitement d'un batch : {e}")
        return [""] * len(batch_prompts)

# Traiter les prompts de plusieurs fichiers
def process_prompts_multi_files(article_prompts, max_batch_size):
    all_results = {file_name: [] for file_name in article_prompts.keys()}
    batch_prompts = []
    batch_metadata = []

    for file_name, prompts in article_prompts.items():
        for prompt_data in prompts:
            if len(batch_prompts) < max_batch_size:
                batch_prompts.append(generate_resume_prompt(prompt_data["question"], prompt_data["retrieved_sections"]))
                batch_metadata.append((file_name, prompt_data))
            else:
                outputs = process_global_batch(batch_prompts)
                for meta, output in zip(batch_metadata, outputs):
                    file_name, row_data = meta
                    row_data["sections_resumees"] = output
                    all_results[file_name].append(row_data)
                batch_prompts = [generate_resume_prompt(prompt_data["question"], prompt_data["retrieved_sections"])]
                batch_metadata = [(file_name, prompt_data)]

    if batch_prompts:
        outputs = process_global_batch(batch_prompts)
        for meta, output in zip(batch_metadata, outputs):
            file_name, row_data = meta
            row_data["sections_resumees"] = output
            all_results[file_name].append(row_data)

    return all_results

# Fonction principale pour traiter plusieurs fichiers
def process_all_resumes(input_paths, chemin_resultats_sources, chemin_dossier_rapport_embeddings):
    os.makedirs(chemin_resultats_sources, exist_ok=True)

    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    default_report_name = "AR6 Climate Change 2022 Mitigation of Climate Change"
    rapport_embeddings = generer_embeddings_rapport(
        os.path.join(chemin_dossier_rapport_embeddings, f"{default_report_name}.json"), embed_model
    )
    embeddings_rapport, sections_rapport, _ = rapport_embeddings

    article_prompts = prepare_prompts_for_files(input_paths, embed_model, embeddings_rapport, sections_rapport)

    max_batch_size = calculate_dynamic_batch_size()
    all_results = process_prompts_multi_files(article_prompts, max_batch_size)

    for file_path, rows in all_results.items():
        if not rows:
            print(f"[WARNING] Aucun résultat pour le fichier : {file_path}")
            continue
        df = pd.DataFrame(rows)
        output_path = os.path.join(chemin_resultats_sources, os.path.basename(file_path).replace("_with_questions.csv", "_resume_sections_results.csv"))
        df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans : {output_path}")

# Finaliser NVML
nvmlShutdown()
