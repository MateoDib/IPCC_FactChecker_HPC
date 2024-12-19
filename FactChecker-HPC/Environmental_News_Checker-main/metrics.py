import os
import pandas as pd
from tqdm import tqdm
import torch
import psutil
from pipeline import pipe  # Pipeline pour le LLM préchargé et partagé horizontalement

# Prompts pour les métriques
prompts = {
    "accuracy": """
        Vous êtes chargé de comparer un extrait d'un article de presse aux informations officielles du rapport du GIEC.
        Évaluez l'exactitude de cet extrait en fonction des sections du rapport fournies. Utilisez l'échelle suivante :

        - **Score 0** : Totalement inexact, contredit directement les informations du rapport.
        - **Score 1** : Très inexact, contient plusieurs erreurs ou omissions importantes par rapport au rapport.
        - **Score 2** : Partiellement inexact, quelques erreurs importantes, mais certaines informations sont correctes.
        - **Score 3** : Modérément exact, contient des informations correctes mais avec quelques imprécisions.
        - **Score 4** : Largement exact, avec seulement de légères imprécisions.
        - **Score 5** : Parfaitement exact, correspond pleinement aux informations du rapport.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """,
    "bias": """
        Vous êtes chargé d'analyser un extrait d'un article de presse pour détecter tout biais potentiel en fonction des sections du rapport du GIEC. Utilisez l'échelle suivante :

        - **Score 0** : Fortement biaisé, présente une vision totalement exagérée ou minimisée.
        - **Score 1** : Biaisé, avec une inclinaison évidente, soit en exagérant soit en minimisant les faits.
        - **Score 2** : Modérément biaisé, certains aspects exagérés ou minimisés mais dans l'ensemble équilibré.
        - **Score 3** : Légèrement biaisé, de petites nuances de biais mais globalement équilibré.
        - **Score 4** : Largement neutre, avec très peu de biais.
        - **Score 5** : Totalement neutre, sans aucun biais détectable.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """,
    "tone": """
        Vous êtes chargé d'analyser le ton d'un extrait d'un article de presse en le comparant aux informations du rapport du GIEC. Utilisez l'échelle suivante :

        - **Score 0** : Ton fortement alarmiste ou minimisant, très éloigné du ton neutre.
        - **Score 1** : Ton exagérément alarmiste ou minimisant.
        - **Score 2** : Ton quelque peu alarmiste ou minimisant.
        - **Score 3** : Ton modérément factuel avec une légère tendance à l'alarmisme ou à la minimisation.
        - **Score 4** : Ton largement factuel, presque totalement neutre.
        - **Score 5** : Ton complètement neutre et factuel, sans tendance perceptible.

        **Extrait de l'article** :
        "{current_phrase}"

        **Sections du rapport du GIEC** :
        "{sections_resumees}"

        Répondez en JSON:
        {{
            "score": <score_entre_0_et_5>,
            "justifications": "<votre_justification>"
        }}
    """
}

# Générer un prompt formaté pour une métrique
def generate_metric_prompt(metric, current_phrase, sections_resumees):
    return prompts[metric].format(current_phrase=current_phrase, sections_resumees=sections_resumees)

# Préparer les prompts pour plusieurs fichiers
def prepare_prompts_from_files(input_paths):
    """Prépare les prompts pour toutes les métriques de chaque fichier."""
    all_prompts = {}

    # Identifier les fichiers
    if isinstance(input_paths, str):  # Si un répertoire est fourni
        file_paths = [os.path.join(input_paths, f) for f in os.listdir(input_paths) if f.endswith("_rag_results.csv")]
    elif isinstance(input_paths, list):  # Si une liste de fichiers est fournie
        file_paths = input_paths
    else:
        raise ValueError("input_paths doit être un chemin de répertoire ou une liste de fichiers.")

    for file_path in tqdm(file_paths, desc="Préparation des prompts"):
        df = pd.read_csv(file_path)

        # Vérifier les colonnes nécessaires
        if not {"current_phrase", "sections_resumees"}.issubset(df.columns):
            print(f"[WARNING] Colonnes manquantes dans le fichier : {file_path}")
            continue

        for metric in prompts.keys():
            df[f"{metric}_prompt"] = df.apply(
                lambda row: generate_metric_prompt(metric, row["current_phrase"], row["sections_resumees"]), axis=1
            )
        all_prompts[file_path] = df
    return all_prompts

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
    
    max_batch_size = max(1, min(max_batch_size, 1000))
    
    print(max_batch_size)

    return max_batch_size

# Traiter un batch global
def process_batch_global(prompts, batch_size):
    """Traite les prompts par batch global."""
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Traitement des prompts"):
        batch_prompts = prompts[i:i + batch_size]
        try:
            outputs = pipe(batch_prompts, max_new_tokens=500, temperature=0.7, batch_size=batch_size, return_full_text=False)
            results.extend([output[0]["generated_text"] for output in outputs])
        except Exception as e:
            print(f"[ERROR] Erreur lors du traitement d'un batch : {e}")
            results.extend([""] * len(batch_prompts))
        torch.cuda.empty_cache()
    return results

# Redistribuer les résultats dans les fichiers d'origine
def redistribute_results(all_prompts, results):
    """Redistribue les résultats dans les fichiers d'origine."""
    index = 0
    for file_path, df in all_prompts.items():
        for metric in prompts.keys():
            num_prompts = len(df)
            df[f"{metric}_response"] = results[index:index + num_prompts]
            index += num_prompts
        output_path = file_path.replace("_rag_results.csv", "_metrics_results.csv")
        df.to_csv(output_path, index=False)
        print(f"Résultats sauvegardés dans : {output_path}")

# Processus principal
def process_all_files(input_paths):
    """Traite plusieurs fichiers avec un batch global."""
    all_prompts = prepare_prompts_from_files(input_paths)
    batch_size = calculate_dynamic_batch_size()

    # Fusionner tous les prompts
    merged_prompts = []
    for df in all_prompts.values():
        for metric in prompts.keys():
            merged_prompts.extend(df[f"{metric}_prompt"].tolist())

    # Traiter les prompts par batch global
    results = process_batch_global(merged_prompts, batch_size)

    # Redistribuer les résultats
    redistribute_results(all_prompts, results)
