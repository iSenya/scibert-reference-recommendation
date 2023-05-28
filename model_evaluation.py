import json, requests
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from preprocessing import strip_all_entities, strip_links
from ModelForEval import ModelForEval
import matplotlib.pyplot as plt
import numpy as np


with open("cleaned_references.json", "r") as f:
    cleaned_references = json.load(f)

print(len(cleaned_references))

model_path = '/Users/senyaisavnina/Downloads/thesis/scibert-reference-recommendation/logs/lightning_logs/version_14/checkpoints/epoch=19-step=8220.ckpt'
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

# # Here I need to retrieve the abstract that my model didn't see

url = "https://api.semanticscholar.org/graph/v1/paper/"
attributes = "paperId,title,abstract"
paperId = "bdf7bf9e81a6c12e22323d0402885b2ba62f623e"
query = f"{url}{paperId}?fields={attributes}"
resp = requests.get(query)

if resp.status_code == 200:
    data = resp.json()
    extracted_data = [
        {
            "paperId": data.get("paperId", ""),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", "")
        }
    ]
else:
    print(resp)

abstract = extracted_data[0]["abstract"]
#abstract = "Flux ropes ejected from the Sun may change their geometrical orientation during their evolution, which directly affects their geoeffectiveness. Therefore, it is crucial to understand how solar flux ropes evolve in the heliosphere to improve our space-weather forecasting tools. We present a follow-up study of the concepts described by Isavnin, Vourlidas, and Kilpua (Solar Phys. 284, 203, 2013). We analyze 14 coronal mass ejections (CMEs), with clear flux-rope signatures, observed during the decay of Solar Cycle 23 and rise of Solar Cycle 24. First, we estimate initial orientations of the flux ropes at the origin using extreme-ultraviolet observations of post-eruption arcades and/or eruptive prominences. Then we reconstruct multi-viewpoint coronagraph observations of the CMEs from ≈ 2 to 30 R⊙ with a three-dimensional geometric representation of a flux rope to determine their geometrical parameters. Finally, we propagate the flux ropes from ≈ 30 R⊙ to 1 AU through MHD-simulated background solar wind while using in-situ measurements at 1 AU of the associated magnetic cloud as a constraint for the propagation technique. This methodology allows us to estimate the flux-rope orientation all the way from the Sun to 1 AU. We find that while the flux-ropes’ deflection occurs predominantly below 30 R⊙, a significant amount of deflection and rotation happens between 30 R⊙ and 1 AU. We compare the flux-rope orientation to the local orientation of the heliospheric current sheet (HCS). We find that slow flux ropes tend to align with the streams of slow solar wind in the inner heliosphere. During the solar-cycle minimum the slow solar-wind channel as well as the HCS usually occupy the area in the vicinity of the solar equatorial plane, which in the past led researchers to the hypothesis that flux ropes align with the HCS. Our results show that exceptions from this rule are explained by interaction with the Parker-spiraled background magnetic field, which dominates over the magnetic interaction with the HCS in the inner heliosphere at least during solar-minimum conditions."
print(abstract)
preprocessed_abstract = strip_all_entities(strip_links(str(abstract)))

# # Tokenize and encode the new abstract
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased", max_model_length = 512)
encoded_inputs = tokenizer(preprocessed_abstract, truncation=True, return_tensors='pt')
input_ids = encoded_inputs['input_ids']


# Extract the embedding vector
with torch.no_grad():
     my_embedding_vector= my_model(input_ids)

my_embedding_tensor = torch.tensor(my_embedding_vector[0])
# Print the embedding vector
print("Embedding Tensor:", my_embedding_tensor)

similarity_scores = []
for ref in cleaned_references:
    ref_embedding_vector = ref["embedding_vector"]
    ref_embedding_tensor = torch.tensor(ref_embedding_vector)
    similarity_score = torch.dot(my_embedding_tensor, ref_embedding_tensor)
    similarity_scores.append((ref["title"], similarity_score.item()))

# Sort the similarity scores in descending order
similarity_scores.sort(key=lambda x: x[1], reverse=True)

# Normalize the similarity scores between 0 and 1
max_score = max(similarity_scores, key=lambda x: x[1])[1]
min_score = min(similarity_scores, key=lambda x: x[1])[1]

normalized_scores = [(title, (score - min_score) / (max_score - min_score)) for title, score in similarity_scores]

true_ref_titles = [ref['title'] for ref in cleaned_references[:33]]

# Separate the scores and titles
scores = [score for _, score in normalized_scores]

# Create a list of indices to represent the position of each score
indices = np.arange(len(scores))

# Plot the histogram for other references
plt.hist(indices, bins=len(scores), weights=scores, color='skyblue', edgecolor='black', label='Other References')

# Highlight the true references
true_ref_indices = [i for i, (title, _) in enumerate(normalized_scores) if title in true_ref_titles]
plt.hist([indices[i] for i in true_ref_indices], bins=len(scores), weights=[scores[i] for i in true_ref_indices], color='pink', edgecolor='black', label='True References')

# Set the y-axis label
plt.ylabel('Normalized Similarity Score')

# Add a legend
plt.legend()

# Adjust the layout to prevent overlapping of bars
plt.tight_layout()

# Show the plot
plt.show()

# # Plot the histogram
# plt.hist(scores, bins=100, color='skyblue', edgecolor='black', density=True)

# # Highlight the true references
# true_ref_indices = [i for i, title in enumerate(titles) if title in true_ref_titles]
# plt.hist([scores[i] for i in true_ref_indices], bins=10, color='pink', edgecolor='black', density=True, alpha=0.5)

# # Set the labels and title
# plt.xlabel('Normalized Similarity Score')
# plt.ylabel('Frequency')
# plt.title('Histogram of Normalized Similarity Scores')

# # Show the plot
# plt.show()


# # Print the top 10 titles
# # Print the top 10 predictions
# print("Top 40 predictions:")
# for title, score in similarity_scores[:40]:
#     print(title, score)

# # Print the prediction with the minimum score
# print("Prediction with the minimum score:")
# min_title, min_score = min(similarity_scores, key=lambda x: x[1])
# print(min_title, min_score)


# for k in [200, 500]:
#     true_pos = 0
#     for title, score in similarity_scores[:k]:
#         if title in true_ref_titles:
#             true_pos += 1
#     print(f'Top-{k} precision = {true_pos/k}, total true refs in range = {true_pos}')
# print(f'Overall distribution = {33/len(cleaned_references)}')


