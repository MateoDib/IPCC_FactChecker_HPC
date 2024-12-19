import pandas as pd
import re
from dateutil import parser

# Dictionary to map French month names to English
french_to_english_months = {
    "janvier": "January", "février": "February", "mars": "March",
    "avril": "April", "mai": "May", "juin": "June",
    "juillet": "July", "août": "August", "septembre": "September",
    "octobre": "October", "novembre": "November", "décembre": "December"
}

# Function to replace French month names with English equivalents
def translate_french_months(date_str):
    for french_month, english_month in french_to_english_months.items():
        date_str = re.sub(french_month, english_month, date_str, flags=re.IGNORECASE)
    return date_str

# Function to remove French day names and parse dates
def parse_date(date_str):
    # List of French day names
    jours_semaine = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    
    # Remove the day of the week if present at the beginning of the string
    for jour in jours_semaine:
        date_str = re.sub(rf"^{jour}\s+", "", date_str, flags=re.IGNORECASE)
    
    # Translate French month names to English
    date_str = translate_french_months(date_str)
    
    # Try parsing the date now that it's in English format
    try:
        parsed_date = parser.parse(date_str, dayfirst=True)
        return parsed_date
    except Exception as e:
        print(f"Could not parse date: {date_str} - {e}")
        return pd.NaT  # If parsing fails, return NaT (Not a Time)

# Main function to process metadata and add the latest GIEC report
def process_metadata_with_giec_reports():
    # Load the metadata file and apply the custom date parsing function
    metadata = pd.read_csv("Data/Index/metadata.csv")
    metadata["Date"] = metadata["Date"].apply(parse_date)  # Convert the 'Date' column to datetime format

    # Check if any dates could not be converted
    if metadata["Date"].isnull().any():
        print("Some dates in 'metadata_sample.csv' could not be converted:")
        print(metadata[metadata["Date"].isnull()]["Date"])

    # Load the GIEC chronology file
    giec_chronologie = pd.read_csv("Data/Index/GIEC_chronologie.csv", sep=";", parse_dates=["date_parution"])

    # Initialize a new column "rapport_GIEC" with empty values
    metadata["rapport_GIEC"] = None

    # Iterate through each row in metadata to find the most recent report
    for i, row in metadata.iterrows():
        if pd.isnull(row["Date"]):
            continue  # Skip rows where the date couldn't be parsed
        
        # Filter GIEC reports published before the article's date
        rapports_avant_date = giec_chronologie[giec_chronologie["date_parution"] <= row["Date"]]
        
        # If there are reports before the date, get the most recent one
        if not rapports_avant_date.empty:
            dernier_rapport = rapports_avant_date.sort_values("date_parution", ascending=False).iloc[0]
            metadata.at[i, "rapport_GIEC"] = dernier_rapport["nom_rapport"]

    # Save the result to a new file
    metadata.to_csv("Data/Index/metadata_with_GIEC.csv", index=False)
    print("The 'rapport_GIEC' column has been added, and the file has been saved.")

# Create the Metadata with GIEC reports
process_metadata_with_giec_reports()
