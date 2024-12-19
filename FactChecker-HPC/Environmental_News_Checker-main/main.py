#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour le traitement d'articles de presse et de rapports du GIEC.
"""

import os
import argparse
from tqdm import tqdm

# Fonction pour enregistrer l'état des ressources
def log_system_resources(stage):
    print(f"\n=== {stage} ===")
    os.system("nvidia-smi")
    os.system("free -h")


# Étape 1 : Nettoyage des articles bruts
def clean_raw_articles():
    from nettoyer_articles import process_all_files_multi_articles
    """
    Étape 1 : Nettoyage des fichiers bruts.
    """
    log_system_resources("Avant nettoyage des articles bruts")

    input_text_dir = "/home2020/home/beta/aebeling/Data/presse/articles_brutes"
    output_text_dir = "/home2020/home/beta/aebeling/Data/presse/articles"

    process_all_files_multi_articles(input_text_dir, output_text_dir)

    log_system_resources("Après nettoyage des articles bruts")


# Étape 2 : Nettoyage des articles pour l'analyse
def clean_press_articles():
    from txt_manipulation import pretraiter_article
    """
    Étape 2 : Nettoyage des articles pour l'analyse.
    """
    log_system_resources("Avant nettoyage des articles analysés")

    chemin_articles = 'Data/presse/articles/'
    chemin_dossier_nettoye = 'Data/presse/articles_cleaned/'
    if not os.path.exists(os.path.dirname(chemin_dossier_nettoye)):
        os.makedirs(os.path.dirname(chemin_dossier_nettoye))

    fichiers_articles = [f for f in os.listdir(chemin_articles) if f.endswith('.txt')]

    for fichier in fichiers_articles:
        chemin_article = os.path.join(chemin_articles, fichier)
        chemin_article_nettoye = os.path.join(chemin_dossier_nettoye, fichier.replace('.txt', '_cleaned.txt'))
        pretraiter_article(chemin_article, chemin_article_nettoye, chemin_dossier_nettoye)

    log_system_resources("Après nettoyage des articles analysés")


# Étape 3 : Nettoyage et indexation des rapports du GIEC
def process_ipcc_reports():
    from pdf_processing import process_pdf_to_index
    """
    Étape 3 : Nettoyage et indexation des rapports du GIEC.
    """
    log_system_resources("Avant indexation des rapports GIEC")

    chemin_rapports_pdf = 'Data/IPCC/rapports/'
    chemin_output_indexed = 'Data/IPCC/rapports_indexed/'

    if not os.path.exists(os.path.dirname(chemin_output_indexed)):
        os.makedirs(os.path.dirname(chemin_output_indexed))

    fichiers_rapports = [f for f in os.listdir(chemin_rapports_pdf) if f.endswith('.pdf')]

    for fichier in fichiers_rapports:
        chemin_rapport_pdf = os.path.join(chemin_rapports_pdf, fichier)
        chemin_rapport_indexed = os.path.join(chemin_output_indexed, fichier.replace('.pdf', '.json'))
        process_pdf_to_index(chemin_rapport_pdf, chemin_rapport_indexed)

    log_system_resources("Après indexation des rapports GIEC")


# Étape 4 : Extraction des références au GIEC
def extract_relevant_ipcc_references():
    from filtrer_extraits import process_all_files
    """
    Étape 4 : Identification des extraits relatifs au GIEC.
    """
    log_system_resources("Avant extraction des références au GIEC")

    chemin_articles_nettoyes = 'Data/presse/articles_cleaned/'
    chemin_output_chunked = '/Data/presse/articles_chunked/'

    os.makedirs(chemin_output_chunked, exist_ok=True)

    fichiers_articles_nettoyes = [
        os.path.join(chemin_articles_nettoyes, f)
        for f in os.listdir(chemin_articles_nettoyes)
        if f.endswith('_processed_cleaned_analysis_results.csv')
    ]

    process_all_files(fichiers_articles_nettoyes, chemin_output_chunked)

    log_system_resources("Après extraction des références au GIEC")


# Étape 5 : Génération des questions
def generate_questions():
    from questions import process_all_files
    """
    Étape 5 : Génération des questions.
    """
    log_system_resources("Avant génération des questions")

    chemin_articles_chunked = 'Data/presse/articles_chunked/'
    chemin_output_questions = 'Data/resultats/resultats_intermediaires/questions/'

    os.makedirs(chemin_output_questions, exist_ok=True)

    # Collecter les fichiers d'entrée
    fichiers_analysis_results = [
        os.path.join(chemin_articles_chunked, f)
        for f in os.listdir(chemin_articles_chunked)
        if f.endswith('_processed_cleaned_analysis_results.csv')
    ]

    print(f"[INFO] Nombre de fichiers trouvés pour traitement : {len(fichiers_analysis_results)}")
    
    if not fichiers_analysis_results:
        print("[WARNING] Aucun fichier valide trouvé pour la génération des questions.")
        return

    # Passer les fichiers au module de traitement
    process_all_files(fichiers_analysis_results, chemin_output_questions)

    log_system_resources("Après génération des questions")



# Étape 6 : Résumé des sources
def summarize_source_sections():
    from resume import process_all_resumes
    """
    Étape 6 : Résumé des sources.
    """
    log_system_resources("Avant résumé des sources")

    chemin_csv_questions = 'Data/resultats/resultats_intermediaires/questions/'
    chemin_resultats_sources = 'Data/resultats/resultats_intermediaires/sources_resumees/'
    dossier_rapport_embeddings = 'Data/IPCC/rapports_indexed/'

    os.makedirs(chemin_resultats_sources, exist_ok=True)

    # Construire les chemins des fichiers à partir du répertoire des questions
    fichiers_questions = [
        os.path.join(chemin_csv_questions, f)
        for f in os.listdir(chemin_csv_questions)
        if f.endswith('_with_questions.csv')
    ]

    # Appeler la fonction principale pour générer les résumés
    process_all_resumes(
        input_paths=fichiers_questions,
        chemin_resultats_sources=chemin_resultats_sources,
        chemin_dossier_rapport_embeddings=dossier_rapport_embeddings
    )

    log_system_resources("Après résumé des sources")



# Étape 7 : Génération des réponses RAG
def generate_rag_responses():
    from reponse import process_all_responses
    """
    Étape 7 : Génération des réponses (RAG).
    """
    log_system_resources("Avant génération des réponses RAG")

    chemin_sources_resumees = 'Data/resultats/resultats_intermediaires/sources_resumees/'
    chemin_output_reponses = 'Data/resultats/resultats_intermediaires/reponses/'

    os.makedirs(chemin_output_reponses, exist_ok=True)

    fichiers_sources_resumees = [
        os.path.join(chemin_sources_resumees, f)
        for f in os.listdir(chemin_sources_resumees)
        if f.endswith('_resume_sections_results.csv')
    ]

    process_all_responses(fichiers_sources_resumees, chemin_output_reponses)

    log_system_resources("Après génération des réponses RAG")


# Étape 8 : Évaluation des réponses
def evaluate_generated_responses():
    from metrics import process_all_files
    """
    Étape 8 : Évaluation des réponses.
    """
    log_system_resources("Avant évaluation des réponses")

    # Définir les chemins des fichiers à évaluer et des résultats
    chemin_reponses = 'Data/resultats/resultats_intermediaires/reponses/'
    chemin_output_evaluation = 'Data/resultats/resultats_intermediaires/evaluation/'

    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(chemin_output_evaluation, exist_ok=True)

    # Récupérer les fichiers à évaluer
    fichiers_reponses = [
        os.path.join(chemin_reponses, f)
        for f in os.listdir(chemin_reponses)
        if f.endswith('_rag_results.csv')
    ]

    # Vérification si des fichiers à évaluer sont présents
    if not fichiers_reponses:
        print("[ERROR] Aucun fichier '_rag_results.csv' trouvé pour l'évaluation des réponses.")
        return

    print(f"[INFO] Nombre de fichiers trouvés pour évaluation : {len(fichiers_reponses)}")

    # Appeler la fonction principale pour traiter les fichiers
    try:
        process_all_files(fichiers_reponses)
        print("[INFO] Évaluation des réponses terminée avec succès.")
    except Exception as e:
        print(f"[ERROR] Erreur lors de l'évaluation des réponses : {e}")

    log_system_resources("Après évaluation des réponses")




# Étape 9 : Parsing des résultats d'évaluation
def parse_evaluation_results():
    from Parsing_exactitude_ton_biais import parsing_all_metrics
    """
    Étape 9 : Parsing des résultats d'évaluation.
    """
    log_system_resources("Avant parsing des résultats d'évaluation")

    input_directory = 'Data/resultats/resultats_intermediaires/evaluation/'
    output_directory = 'Data/resultats/resultats_finaux/resultats_csv/'

    os.makedirs(output_directory, exist_ok=True)

    parsing_all_metrics(input_directory, output_directory)

    log_system_resources("Après parsing des résultats d'évaluation")


# Étape 10 : Conversion en JSON
def results_to_json():
    from Structure_JSON import structurer_json
    """
    Étape 10 : Conversion des résultats en JSON.
    """
    log_system_resources("Avant conversion des résultats en JSON")

    evaluation_dir = 'Data/resultats/resultats_finaux/resultats_csv/'
    article_dir = 'Data/presse/articles_chunked/'
    output_dir = 'Data/resultats/resultats_finaux/resultats_json/'

    os.makedirs(output_dir, exist_ok=True)

    structurer_json(evaluation_dir, article_dir, output_dir)

    log_system_resources("Après conversion des résultats en JSON")


# Étape 11 : Création du HTML
def html_visualisation_creation():
    from Creation_code_HTML import generate_html_from_json
    """
    Étape 11 : Création du HTML pour la visualisation des résultats.
    """
    log_system_resources("Avant création du HTML pour visualisation")

    json_dir = "Data/resultats/resultats_finaux/resultats_json/"
    output_html = "Data/resultats/Visualisation_results.html"
    articles_data_dir = "articles_data/"

    generate_html_from_json(json_dir, output_html, articles_data_dir)

    log_system_resources("Après création du HTML pour visualisation")


# Pipeline complète
def run_full_processing_pipeline():
    """
    Exécute toutes les étapes dans l'ordre.
    """
    log_system_resources("Début de la pipeline complète")

    clean_raw_articles()
    clean_press_articles()
    process_ipcc_reports()
    extract_relevant_ipcc_references()
    generate_questions()
    summarize_source_sections()
    generate_rag_responses()
    evaluate_generated_responses()
    parse_evaluation_results()
    results_to_json()
    html_visualisation_creation()

    log_system_resources("Fin de la pipeline complète")


# Interface CLI
def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline principale pour le traitement des articles et rapports du GIEC.")
    parser.add_argument("--task", type=int, default=11, help="Choisir quelle étape exécuter (1 à 11).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    choice = args.task

    match str(choice):
        case "1":
            clean_raw_articles()
        case "2":
            clean_press_articles()
        case "3":
            process_ipcc_reports()
        case "4":
            extract_relevant_ipcc_references()
        case "5":
            generate_questions()
        case "6":
            summarize_source_sections()
        case "7":
            generate_rag_responses()
        case "8":
            evaluate_generated_responses()
        case "9":
            parse_evaluation_results()
        case "10":
            results_to_json()
        case "11":
            run_full_processing_pipeline()
        case _:
            print("Option invalide. Veuillez choisir un numéro d'étape valide.")
