#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 04:35:55 2024

@author: mateodib
"""

import pandas as pd
import re
import os

# Function to manually extract 'score' and 'justifications' from simulated JSON text
def parse_simulated_json(response_text):
    """
    Parses simulated JSON-style text and extracts 'score' and 'justifications'.
    Returns None for both if parsing fails.
    """
    # Initialize the result with None values
    score = None
    justification = None
    
    if pd.isnull(response_text):
        return score, justification
    
    # Extract score using regex
    score_match = re.search(r'"score"\s*:\s*(\d+)', response_text)
    if score_match:
        score = int(score_match.group(1))
    
    # Extract justification using regex
    justification_match = re.search(r'"justifications"\s*:\s*"(.*?)"', response_text, re.DOTALL)
    if justification_match:
        justification = justification_match.group(1).strip()
    
    return score, justification

# Process each CSV file in the input directory
def parsing_all_metrics(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".csv"):
            # Construct the full file path
            chemin_fichier = os.path.join(input_directory, filename)
            
            # Load the data from the CSV file
            df = pd.read_csv(chemin_fichier)

            # Initialize a list to store the parsed data
            parsed_data = []

            # Identify dynamically the metrics columns in the dataset
            metric_columns = [col for col in df.columns if col not in ['id', 'question', 'current_phrase', 'sections_resumees']]
            
            # Iterate through each row to extract the information
            for _, row in df.iterrows():
                # Extract information from each metric dynamically
                parsed_row = {
                    'id': row['id'],
                    'question': row['question'],
                    'current_phrase': row['current_phrase'],
                    'sections_resumees': row['sections_resumees']
                }
                
                for metric in metric_columns:
                    score, justification = parse_simulated_json(row[metric])
                    parsed_row[f'{metric}_score'] = score
                    parsed_row[f'{metric}_justification'] = justification
                
                # Append the extracted data to the list
                parsed_data.append(parsed_row)

            # Convert the parsed data into a DataFrame for saving
            df_parsed = pd.DataFrame(parsed_data)

            # Define the output file path based on the original file name
            output_filename = f"{os.path.splitext(filename)[0]}_parsed.csv"
            chemin_sortie = os.path.join(output_directory, output_filename)
            
            # Save the parsed DataFrame to a new CSV file
            df_parsed.to_csv(chemin_sortie, index=False, quotechar='"')
            print(f"Parsed data saved in {chemin_sortie}")