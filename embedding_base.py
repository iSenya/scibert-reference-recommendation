import json
import requests
import langdetect
from preprocessing import strip_links, strip_all_entities
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from ModelForEval import ModelForEval

config_path = '/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/logs/lightning_logs/version_14/hparams.yaml'
model_path = '/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/logs/lightning_logs/version_14/checkpoints/epoch=19-step=8220.ckpt'
config = AutoConfig.from_pretrained(config_path)
bert_model = AutoModel.from_config(config)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))["state_dict"]
filtered_state_dict = {k: v for k, v in state_dict.items() if k in bert_model.state_dict()}
bert_model.load_state_dict(filtered_state_dict, strict=False)

file_paths = [
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations.json",
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations_1.json",
    "/Users/senyaisavnina/Downloads/extracted_data_w_citations_2.json",
]

full_dataset = []

for path in file_paths:
    with open(path, "r") as f:
        dataset = json.load(f)
        full_dataset.extend(dataset)

url = "https://api.semanticscholar.org/graph/v1/paper/"
paperId = "bdf7bf9e81a6c12e22323d0402885b2ba62f623e"
attributes = "paperId,title,abstract,references.title,references.abstract,references.paperId"

query = f"{url}{paperId}?fields={attributes}"
resp = requests.get(query)

if resp.status_code == 200:
    data = resp.json()
    extracted_data = [
        {
            "paperId": data.get("paperId", ""),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "references": data.get("references", [])
        }
    ]
else:
    print(resp)

cleaned_references = []

# Create an instance of the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")

for data in extracted_data:
    references = data["references"]

    # Iterate over the references
    for reference in references:
        paper_id = reference.get("paperId", "")
        title = reference.get("title", "")
        abstract = reference.get("abstract", "")

        # Check if the paper ID exists in the full dataset
        if paper_id not in [item["paperId"] for item in full_dataset]:
            # Check if the abstract is not None, a non-empty string, and in English
            if abstract is not None and isinstance(abstract, str) and langdetect.detect(abstract) == "en":
                # Clean the abstract
                cleaned_abstract = strip_all_entities(strip_links(str(abstract)))

                # Tokenize and encode the cleaned abstract
                encoded_inputs = tokenizer(cleaned_abstract, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
                input_ids = encoded_inputs['input_ids']
                input_ids = input_ids[:, :torch.nonzero(input_ids[0] != 0).max() + 1]

                # Create an instance of the BERT model
                my_model = ModelForEval(bert_model)

                # Extract the embedding vector
                with torch.no_grad():
                    embedding_vector = my_model(input_ids)

                # Append the paper ID, title, and embedding vector to the list
                cleaned_references.append({
                    "paperId": paper_id,
                    "title": title,
                    "embedding_vector": embedding_vector.tolist()[0]
                })

# Save the cleaned references to a JSON file
output_file = "cleaned_references.json"
with open(output_file, "w") as f:
    json.dump(cleaned_references, f, indent=4)

print("Cleaned references saved to", output_file)