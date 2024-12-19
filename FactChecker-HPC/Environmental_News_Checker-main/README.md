# Environmental News Checker

This repository provides a comprehensive toolkit for validating and analyzing environmental news articles using IPCC reports as a reference. The project aims to facilitate fact-checking by evaluating the alignment of news content with scientific data from IPCC reports, assessing accuracy, tone, and potential biases within the articles.

## Objective

The primary goal of this project is to create a reliable, automated way to cross-reference environmental news with authoritative IPCC reports. This tool is intended for journalists, researchers, and fact-checkers who want to ensure that environmental reporting aligns with scientific consensus and accurately reflects current knowledge on climate issues.

## Methodology

The Environmental News Checker operates through a multi-step pipeline:

1. Text Parsing and Analysis: News articles are parsed to extract relevant content segments, which are then analyzed based on predefined criteria.
2. Question Generation and Validation: For each article, relevant questions are generated to probe the scientific validity of its claims. These questions are then used to query IPCC reports for verification.
3. Response Analysis: Retrieved information is compared against the article content, assessing its alignment with IPCC data.
4. Summarization and Reporting: The tool generates a summary report highlighting the accuracy, tone, and any biases found in the article, along with specific references to relevant sections in IPCC reports.

## Flexibility of Implementation

The project can be run entirely using open-source code or by leveraging external APIs, such as the Replicate API, to enhance the analysis process. Users can choose between these options depending on their resources and preference for open-source tools versus API-driven solutions.

### Open-Source Option

All core functionalities are available as open-source Python scripts, enabling users to run the tool locally without the need for external API access.

### API Integration

For enhanced processing power and model access, this tool can integrate with the Replicate API. This option allows for more sophisticated and scalable language model interactions, particularly useful for complex natural language processing tasks.

## Getting Started

### Prerequisites

Ensure that you have Python 3.6 or later installed, along with the necessary Python libraries. Install dependencies with:

```{python}
pip install -r requirements.txt
```


### Clone the Repository

```{python}
git clone https://github.com/your-username/environmental-news-checker.git
cd environmental-news-checker
```


### Running the Tool

To start the analysis, execute the main script:

```{python}
python main.py
```

This command initiates the entire processing pipeline, from parsing and question generation to response validation and summarization.


## Installion Llama3.2 en local 

### Prérequis

Assurez-vous que vous avez un Mac M2 (ou une machine compatible) car Ollama est optimisé pour les environnements Apple Silicon (M1/M2).

### Étapes d'installation

1. Installer Ollama
Vous pouvez effectuer cette étape simplement en allant sur le lien : https://ollama.com/

2. Télécharger le modèle Llama 3.2
Une fois Ollama installé, vous pouvez télécharger le modèle Llama 3.2 (par exemple, la version 1B ou 3B en fonction de vos besoins). Utilisez la commande suivante :

```{python}
ollama pull llama3.2:3b-instruct-fp16
```

Remplacez 3b par 1b si vous souhaitez télécharger la version 1B du modèle.
Ollama va gérer le téléchargement du modèle en local et le préparer pour une utilisation directe.


3. érifier l'installation
Après le téléchargement, vous pouvez vérifier si le modèle est bien installé et prêt à être utilisé en exécutant la commande suivante :

```{python}
ollama list
```

Il faudra toujours vérifier que l'application Ollama est ouverte avant de faire tourner le code.
