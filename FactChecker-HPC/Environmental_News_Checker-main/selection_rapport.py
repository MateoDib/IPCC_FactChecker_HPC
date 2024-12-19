import pandas as pd
from difflib import SequenceMatcher

# Function to find the report name based on article title
def find_report_by_title(article_title):
    """
    Given an article title, find the associated report name (rapport_GIEC) based on a 90% similarity.
    
    Args:
        article_title (str): The title of the article to search for.
    
    Returns:
        str: The name of the GIEC report associated with the article title, or a default report name if no close match is found.
    """
    # Load the metadata with GIEC information
    metadata_with_giec = pd.read_csv("Data/Index/metadata_with_GIEC.csv")

    # Remove underscores from the "Title" column to improve matching
    metadata_with_giec["Title"] = metadata_with_giec["Title"].str.replace("_", "", regex=False)

    # Set a threshold for similarity
    threshold = 0.9  # 90% similarity

    # Initialize variables to store the best match
    best_match = None
    best_score = 0

    # Iterate over each title in the metadata
    for _, row in metadata_with_giec.iterrows():
        metadata_title = row["Title"].lower()
        
        # Calculate similarity score
        similarity_score = SequenceMatcher(None, metadata_title, article_title.lower()).ratio()
        
        # Check if this match is above the threshold and better than the previous best match
        if similarity_score >= threshold and similarity_score > best_score:
            best_match = row["rapport_GIEC"]
            best_score = similarity_score

    # Return the best match if found, otherwise return the default report
    if best_match:
        return best_match
    else:
        # Return the default report if no match is found
        return "AR6 Climate Change 2022 Mitigation of Climate Change"


