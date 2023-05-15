import requests
import json
from concurrent.futures import ThreadPoolExecutor

url = "https://api.semanticscholar.org/graph/v1/paper/"
paperId = "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
attributes = "paperId,title,abstract,references.abstract,references.paperId,citations.paperId"

def get_paper_data(paper_id):
    try:
        response = requests.get(f"{url}{paper_id}?fields=title,abstract,references.abstract")
        data = response.json()
        referenced_abstracts = [ref["abstract"] for ref in data["references"]]
        return {
            "paperId": paper_id,
            "title": data["title"],
            "abstract": data["abstract"],
            "referenced_abstracts": referenced_abstracts
        }
    except (KeyError, requests.exceptions.RequestException):
        return None

query = f"{url}{paperId}?fields={attributes}"
resp = requests.get(query)

if resp.status_code == 200:
    
    data = resp.json()

    extracted_data = [
        {
            "paperId": data.get("paperId", ""),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "referenced_abstracts": [ref.get("abstract", "") for ref in data.get("references", [])]
        }
    ]

    citations = [{"paperId": cit.get("paperId", ""), "title": cit.get("title", "")} for cit in data.get("citations", [])]
    
    # retrieving citations if there are more than 1000

    next_cursor = data.get("next_cursor", None)

    while next_cursor is not None:
        cursor_query = f"{url}{paperId}?fields=citations.paperId,next_cursor&cursor={next_cursor}"
        cursor_resp = requests.get(cursor_query)
        cursor_data = cursor_resp.json()
        citations += [{"paperId": cit.get("paperId", ""), "title": cit.get("title", "")} for cit in cursor_data.get("citations", [])]
        next_cursor = cursor_data.get("next_cursor", None)

    references = [{"paperId": ref.get("paperId", ""), "title": ref.get("title", ""), "abstract": ref.get("abstract", "")} for ref in data.get("references", [])]


    paper_ids = [ref["paperId"] for ref in references] + [cit["paperId"] for cit in citations]

    with ThreadPoolExecutor() as executor:
        paper_data = list(executor.map(get_paper_data, paper_ids))
    paper_data = [d for d in paper_data if d is not None]

    extracted_data.extend(paper_data)

    print(json.dumps(extracted_data, indent=2))
else:
    print("Error: ", resp.status_code)

with open("extracted_data_w_citations.json", "w") as f:
  json.dump(extracted_data, f)