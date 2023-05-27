import requests
import time
import langdetect
from preprocessing import strip_links, strip_all_entities
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from ModelForEval import ModelForEval
import json


config_path = '/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/logs/lightning_logs/version_14/hparams.yaml'
model_path = '/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/logs/lightning_logs/version_14/checkpoints/epoch=19-step=8220.ckpt'
config = AutoConfig.from_pretrained(config_path)

paper_ids = set()

checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
state_dict = checkpoint["state_dict"]
bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
my_model = ModelForEval(bert_model)

my_model.load_state_dict(
    {
        "fc1.weight": state_dict["model.fc1.weight"],
        "fc1.bias": state_dict["model.fc1.bias"],
        "fc2.weight": state_dict["model.fc2.weight"],
        "fc2.bias": state_dict["model.fc2.bias"],
    },
    strict=False
)

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


# Create an instance of the tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased", model_max_length=512)

# Retrieve data for the initial paper
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

references = extracted_data[0]["references"]



cleaned_references = []

# Iterate over the references
for reference in references:
    ref_paper_id = reference.get("paperId", "")
    title = reference.get("title", "")
    abstract = reference.get("abstract", "")

    # Check if the paper ID exists in the full dataset
    if ref_paper_id not in [item["paperId"] for item in full_dataset]:
        # Check if the abstract is not None, a non-empty string, and in English
        if abstract is not None and isinstance(abstract, str) and langdetect.detect(abstract) == "en":
            # Clean the abstract
            cleaned_abstract = strip_all_entities(strip_links(str(abstract)))

            # Tokenize and encode the cleaned abstract
            encoded_inputs = tokenizer(cleaned_abstract,
                                       truncation=True, 
                                       return_tensors='pt')
            input_ids = encoded_inputs['input_ids']
            print(input_ids.shape)
            
            # Extract the embedding vector
            with torch.no_grad():
                embedding_vector = my_model(input_ids)

            # Append the paper ID, title, and embedding vector to the list
            if ref_paper_id not in paper_ids:
                cleaned_references.append({
                    "paperId": ref_paper_id,
                    "title": title,
                    "embedding_vector": embedding_vector.tolist()[0]
                })
                paper_ids.add(ref_paper_id)

print(len(cleaned_references))

# Set the desired number of items and initial offset

random_paper_url = "https://api.semanticscholar.org/graph/v1/paper/search"
desired_num_items = 2000
offset = 0
max_retries = 3  # Maximum number of retries
retry_wait_time = 10  # Duration to wait before retrying in seconds
items_per_request = 90
retries = 0
success = False

# Retrieve data for random papers until the desired number of items is reached
while len(cleaned_references) < desired_num_items:
    # Determine the number of random papers to retrieve in this iteration
    remaining_items = desired_num_items - len(cleaned_references)
    num_items_to_retrieve = min(remaining_items, items_per_request)

    # Define the parameters as a dictionary
    params = {
        "query": "and",
        "fields": "title,abstract,references.paperId,references.abstract,references.title",
        "limit": num_items_to_retrieve,
        "offset": offset
    }

    retries = 0
    success = False

    while retries < max_retries and not success:
        # Make the GET request with the updated parameters
        response = requests.get(random_paper_url, params=params)

        if response.status_code == 429:  # Rate limit exceeded
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(retry_wait_time)
            retries += 1
            continue

        try:
            random_papers = response.json()
        except json.JSONDecodeError as e:
            print("Error decoding JSON response:", str(e))

        if not isinstance(random_papers, dict) or "data" not in random_papers:
            print("Error retrieving random papers:", random_papers)
            retries += 1
            time.sleep(10)  # Wait for a short duration before retrying
            continue

        if "error" in random_papers and random_papers["error"] == "Not found":
            print("No random papers found")
            retries += 1
            time.sleep(10)  # Wait for a short duration before retrying
            continue

        success = True

    if not success:
        print("Failed to retrieve random papers. Exiting loop.")
        break
    
    if response.status_code == 200:
        success = True

    for random_paper in random_papers['data']:
        ref_paper_id = random_paper.get("paperId", "")
        title = random_paper.get("title", "")
        abstract = random_paper.get("abstract", "")
        references = random_paper.get("references", [])

        # Check if the abstract is not None, a non-empty string, and in English
        if abstract is not None and isinstance(abstract, str) and len(abstract) > 10 and langdetect.detect(abstract) == "en":
            # Clean the abstract
            cleaned_abstract = strip_all_entities(strip_links(str(abstract)))
            # Tokenize and encode the cleaned abstract
            encoded_inputs = tokenizer(
                cleaned_abstract,
                # padding='max_length',
                truncation=True,
                # max_length=512,
                return_tensors='pt',
            )
            input_ids = encoded_inputs['input_ids']
            print(input_ids.shape)
            # print(cleaned_abstract)

            # Extract the embedding vector
            with torch.no_grad():
                embedding_vector = my_model(input_ids)
            # Append the paper ID, title, and embedding vector to the list
            if ref_paper_id not in paper_ids:
                cleaned_references.append({
                    "paperId": ref_paper_id,
                    "title": title,
                    "embedding_vector": embedding_vector.tolist()[0]
                })
                paper_ids.add(ref_paper_id)

        # Iterate over the references of the random paper
        for reference in references:
            ref_paper_id = reference.get("paperId", "")
            ref_title = reference.get("title", "")
            ref_abstract = reference.get("abstract", "")

            # Check if the reference's abstract is not None, a non-empty string, and in English
            if ref_abstract is not None and len(ref_abstract) > 10 and langdetect.detect(ref_abstract) == "en":
                # Clean the reference's abstract
                cleaned_ref_abstract = strip_all_entities(strip_links(str(ref_abstract)))
                # Tokenize and encode the cleaned reference's abstract
                encoded_inputs = tokenizer(
                    cleaned_ref_abstract,
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoded_inputs['input_ids']
                print(input_ids.shape)
                # print(cleaned_ref_abstract)
                
                # Extract the embedding vector
                with torch.no_grad():
                    embedding_vector = my_model(input_ids)

                # Append the reference's paper ID, title, and embedding vector to the list
                if ref_paper_id not in paper_ids:
                    cleaned_references.append({
                        "paperId": ref_paper_id,
                        "title": ref_title,
                        "embedding_vector": embedding_vector.tolist()[0]
                    })
                    paper_ids.add(ref_paper_id)

    # Update the offset for the next iteration
    offset += num_items_to_retrieve

    # Wait for a short duration to avoid rate limits
    time.sleep(retry_wait_time)


# Save the cleaned references to a JSON file
output_file = "cleaned_references.json"
with open(output_file, "w") as f:
    json.dump(cleaned_references, f, indent=4)
print("Cleaned references saved to", output_file)
