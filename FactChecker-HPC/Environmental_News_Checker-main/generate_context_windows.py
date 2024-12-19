"""
Ce script détecte les termes du glossaire dans un article, génère des fenêtres contextuelles à partir des phrases de l'article,
et compare ces fenêtres avec des sections d'un rapport en utilisant les embeddings. Il ajuste dynamiquement le seuil de similarité et 
génère des mentions lorsque des sections similaires sont trouvées.

Fonctionnalités principales :
- Détection de termes du glossaire dans une phrase
- Génération de fenêtres contextuelles à partir de phrases
- Comparaison de fenêtres contextuelles avec des sections du rapport en utilisant les embeddings
- Détection de termes du glossaire et ajout de définitions correspondantes dans les résultats

"""


def generate_context_windows(phrases, window_size=2, len_phrase_focus=1):
    """
    Génère des fenêtres contextuelles à partir de la liste des phrases.

    Args:
        phrases (list[str]): Liste de phrases à regrouper en fenêtres contextuelles.
        window_size (int): Taille de la fenêtre (nombre de phrases par fenêtre).

    Returns:
        list: Liste des fenêtres contextuelles sous forme de dictionnaires.
    """
    windows = []
    focus_window = []
    for i in range(0, len(phrases), 3):
        # Combine les phrases avant et après la phrase actuelle dans une seule chaîne de caractères
        context_window = " ".join(
            phrases[max(0, i - window_size):min(i + window_size + 1, len(phrases))])
        focus_window = " ".join(
            phrases[max(0, i - len_phrase_focus):min(i + len_phrase_focus + 1, len(phrases))])

        windows.append({
            "id": i,  # Ajout de l'index
            "context": context_window,
            "current_phrase": focus_window
        })
    return windows

