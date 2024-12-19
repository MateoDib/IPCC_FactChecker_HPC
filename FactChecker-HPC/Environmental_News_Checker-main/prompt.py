from langchain import PromptTemplate
from langchain_ollama import OllamaLLM


"""

Functions for extract_relevant_ipcc_references

"""
# Fonction pour créer le prompt pour chaque phrase
def prompt_selection_phrase_pertinente(current_phrase, context):
    return f"""
    Vous êtes un expert chargé d'identifier tous les sujets abordés dans le texte suivant, qu'ils soient ou non liés à l'environnement, au changement climatique ou au réchauffement climatique.
    
    Phrase : {current_phrase}
    context : {context}
    
    1. Si le texte mentionne de près ou de loin l'environnement, le changement climatique, le réchauffement climatique, ou des organisations, événements ou accords liés à ces sujets (par exemple le GIEC, les conférences COP, les accords de Paris, etc.), répondez '1'. Sinon, répondez '0'.
    2. Listez **tous** les sujets abordés dans le texte, y compris ceux qui ne sont pas liés à l'environnement ou au climat.
    
    Format de réponse attendu :
    - Réponse binaire (0 ou 1) : [Réponse]
    - Liste des sujets abordés : [Sujet 1, Sujet 2, ...]
    
    Exemple de réponse :
    - Réponse binaire (0 ou 1) : 1
    - Liste des sujets abordés : [Incendies, gestion des forêts, réchauffement climatique, économie locale, GIEC]
    """




def create_questions_llm(model_name="llama3.2:3b-instruct-fp16"):
    """
    Initialise le modèle LLM et crée une LLMChain.
    """
    llm = OllamaLLM(model=model_name)
    prompt_template = """
    Vous êtes chargé de formuler une **question précise** pour vérifier les informations mentionnées dans un extrait spécifique d'un article de presse en consultant directement les rapports du GIEC (Groupe d'experts intergouvernemental sur l'évolution du climat).

    Cette question sera utilisée dans un système de récupération d'information (RAG) pour extraire les sections pertinentes des rapports du GIEC et comparer les informations des rapports avec celles de l'article de presse.

    **Objectif** : La question doit permettre de vérifier si les informations fournies dans la phrase de l'article sont corroborées ou contestées par les preuves scientifiques dans les rapports du GIEC. La question doit donc englober tous les sujets abordés par l'extrait de l'article de presse.

    **Instructions** :

    1. Analysez l'extrait et son contexte pour identifier les affirmations clées et/ou les informations à vérifier.
    2. Formulez une **question claire et spécifique** orientée vers la vérification de ces affirmations ou informations à partir des rapports du GIEC. La question doit permettre de vérifier toutes les informations de l'extraits. La question peut être un ensemble de questions comme : "Quel est l'impact des activités humaines sur le taux de CO2 dans l'atomsphère ? Comment la concentration du CO2 dans l'atmosphère impact l'argiculture?"
    3. La question doit être **directement vérifiable** dans les rapports du GIEC via un système RAG.
    4. **IMPORTANT** : Répondez uniquement avec la question, sans ajouter d'explications ou de contexte supplémentaire.

    Extrait de l'article de presse : {current_phrase}

    Contexte : {context}

    Générez uniquement la **question** spécifique qui permettrait de vérifier les informations mentionnées dans cette phrase en consultant les rapports du GIEC via un système de récupération d'information (RAG).
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "current_phrase", "context"])

    # Directly chain prompt with LLM using the | operator
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Use pipe to create a chain where prompt output feeds into the LLM
    return llm_chain





def creer_llm_resume(model_name="llama3.2:3b-instruct-fp16"):
    """
    Creates and configures the LLM chain for summarization.
    """
    llm = OllamaLLM(model=model_name)
    prompt_template_resume = """
    **Tâche** : Fournir un résumé détaillé et structuré des faits scientifiques contenus dans la section du rapport du GIEC, en les reliant directement à la question posée. La réponse doit être sous forme de liste numérotée, avec chaque point citant précisément les données chiffrées ou informations textuelles pertinentes.

    **Instructions** :
    - **Objectif** : Présenter une liste complète des faits pertinents, incluant les éléments directement en lien avec la question, ainsi que des informations contextuelles importantes qui peuvent enrichir la compréhension du sujet.
    - **Directives spécifiques** :
        1. Inclure des faits scientifiques directement en rapport avec la question, en les citant de manière précise.
        2. Intégrer des données chiffrées, tendances, et statistiques spécifiques lorsque disponibles, en veillant à la clarté et la précision de la citation.
        3. Fournir des éléments de contexte pertinents qui peuvent éclairer la réponse, sans extrapoler ou interpréter.
        4. Utiliser un langage concis mais précis, en mentionnant chaque fait pertinent dans une ou deux phrases.
        5. Être exhaustif dans les faits listés pouvant être intéressant, directement ou indirectement pou répondre à la question.
    - **Restrictions** : 
        - Ne pas inclure d'opinions ou de généralisations.
        - Ne reformuler les informations que si cela permet de les rendre plus compréhensibles sans en altérer le sens.
        - Ne présenter que des faits, sans ajout de suppositions ou interprétations.
        - Ne pas ajouter de phrase introductrice comme 'Voici les fais retranscris' ou autre.
    - **Format de réponse** : Utiliser une liste numérotée, en commençant par les faits les plus directement liés à la question, suivis par les éléments de contexte. Chaque point doit être limité à une ou deux phrases.

    ### Question :
    "{question}"

    ### Sections du rapport :
    {retrieved_sections}

    **Exemple de réponse attendue** :
        1. Le niveau global de la mer a augmenté de 0,19 m entre 1901 et 2010, en lien direct avec la hausse des températures mondiales.
        2. Les températures moyennes ont augmenté de 1,09°C entre 1850-1900 et 2011-2020, influençant la fréquence des événements climatiques extrêmes.
        3. Les concentrations de CO2 dans l'atmosphère ont atteint 410 ppm en 2019, une donnée clé pour comprendre l'accélération du réchauffement climatique.

    **Remarque** : Respecter strictement les consignes et ne présenter que des faits sous forme de liste numérotée. Citer toutes les données chiffrées ou textuelles de manière exacte pour assurer la rigueur de la réponse.
    """
    prompt = PromptTemplate(template=prompt_template_resume, input_variables=[
                            "question", "retrieved_sections"])

    # Directly chain prompt with LLM using the | operator
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    # Use pipe to create a chain where prompt output feeds into the LLM
    return llm_chain




def creer_prompt_reponses():
    # Initialize LLM with the specified model
    llm = OllamaLLM(model="llama3.2:3b-instruct-fp16")

    # Define the prompt for the LLM
    prompt_template = """
    Vous êtes un expert en climatologie. Répondez à la question ci-dessous en vous basant uniquement sur les sections pertinentes du rapport du GIEC.

    **Instructions** :
    1. Utilisez les informations des sections pour formuler une réponse précise et fondée.
    2. Justifiez votre réponse en citant les sections, si nécessaire.
    3. Limitez votre réponse aux informations fournies dans les sections.

    **Question** : {question}
    
    **Sections du rapport** : {consolidated_text}
    
    **Réponse** :
    - **Résumé de la réponse** : (Réponse concise)
    - **Justification basée sur le rapport** : (Citez et expliquez les éléments pertinents)
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=[
                            "question", "consolidated_text"])
    llm_chain = prompt | llm  # Using simplified chaining without LLMChain

    return llm_chain
