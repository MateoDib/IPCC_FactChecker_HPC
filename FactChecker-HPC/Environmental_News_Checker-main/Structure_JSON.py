import pandas as pd
import json
import os

# Loop through each evaluation file in the directory
def structurer_json(evaluation_dir, article_dir, output_dir):
    for filename in os.listdir(evaluation_dir):
        if filename.endswith("_parsed.csv"):
            # Paths for the evaluation file and output JSON
            chemin_evaluation_parsed = os.path.join(evaluation_dir, filename)
            output_json_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")

            # Load the evaluation data with fact-checked chunks
            df_evaluation_parsed = pd.read_csv(chemin_evaluation_parsed)
            
            # Dynamically detect available metrics by filtering columns with `_score`
            available_metrics = [col.split('_score')[0] for col in df_evaluation_parsed.columns if '_score' in col]
            
            # Determine the corresponding article file path based on the evaluation file name
            article_filename = filename.replace("_evaluation_results_parsed.csv", "_analysis_results.csv")
            chemin_article = os.path.join(article_dir, article_filename)

            # Check if the article file exists
            if not os.path.exists(chemin_article):
                print(f"Article file {chemin_article} not found. Skipping.")
                continue

            # Load the climate analysis data with all chunks of 3 sentences
            df_climate_analysis = pd.read_csv(chemin_article)

            # Merge evaluation data with climate analysis data on "id"
            df_merged = df_climate_analysis.merge(df_evaluation_parsed.drop(columns=['current_phrase']), on="id", how="left")

            # Extract the article title from the file name (remove path and extension)
            article_title = os.path.splitext(article_filename)[0]

            # Initialize a dictionary to store structured JSON data
            structured_data = {
                "article_title": article_title,
                "phrases": {}
            }

            # Iterate over the merged DataFrame to build the structured JSON data
            for _, row in df_merged.iterrows():
                # Prepare analysis data for each metric, using dictionary comprehension
                analysis_data = {
                    metric: {
                        "score": row[f"{metric}_score"] if f"{metric}_score" in row and pd.notna(row[f"{metric}_score"]) else None,
                        "justifications": row[f"{metric}_justification"] if f"{metric}_justification" in row and pd.notna(row[f"{metric}_justification"]) else None
                    }
                    for metric in available_metrics
                }

                # Prepare phrase data
                phrase_data = {
                    "text": row['current_phrase'] if 'current_phrase' in row else None,
                    "context": row['context'] if 'context' in row else None,
                    "analysis": analysis_data
                }

                # Add the phrase data to the JSON structure under its "id" as the key
                structured_data["phrases"][str(row['id'])] = phrase_data

            # Save the structured data as JSON
            with open(output_json_path, 'w', encoding='utf-8') as json_file:
                json.dump(structured_data, json_file, indent=4, ensure_ascii=False)

            print(f"Structured article analysis saved to {output_json_path}")