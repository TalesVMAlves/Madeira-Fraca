import argilla as rg
import pandas as pd
from datasets import Dataset as HFDataset
import os

ARGILLA_API_URL = "http://localhost:6900"
ARGILLA_API_KEY = "argilla.apikey"
ARGILLA_DATASET_NAME = "descricao_madeira_ner"
ARGILLA_WORKSPACE = "default"
OUTPUT_CSV_PATH = "dataset/labeledTokenClassification.csv"

print("Connecting to Argilla...")
client = rg.Argilla(
    api_url=ARGILLA_API_URL,
    api_key=ARGILLA_API_KEY,
)
print("Connection successful.")

try:
    print(f"Fetching dataset '{ARGILLA_DATASET_NAME}' from workspace '{ARGILLA_WORKSPACE}'...")
    dataset = client.datasets(name=ARGILLA_DATASET_NAME, workspace=ARGILLA_WORKSPACE)
    print("Dataset found.")

    print("Exporting records from Argilla...")
    hf_dataset: HFDataset = dataset.records.to_datasets()
    print(f"Exported {len(hf_dataset)} records into a Hugging Face Dataset object.")

    print("Converting to pandas DataFrame...")
    df_exported = hf_dataset.to_pandas()
    print("Conversion successful.")

    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    df_exported.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\nSuccessfully exported dataset records to '{OUTPUT_CSV_PATH}'")
    print("The CSV file contains the original data plus any annotations (responses/suggestions) you added.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure Argilla server is running and the dataset name/workspace are correct.")