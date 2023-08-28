from typing import List
import requests
import copy
import json
import time
import math
import random
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SEMANTIC_SCHOLAR_API_URL = 'https://api.semanticscholar.org/graph/v1/'
API_KEY = 't5z5FBumibq2Z4tlNTGBaM31fzrfDhJ1PirrKTJ4'

headers = {
    'x-api-key': API_KEY
}

def throttled_request(*args, **kwargs):
    response = requests.request(*args, **kwargs)
    if response.status_code == 429:
        time.sleep(10)
        return throttled_request(*args, **kwargs)
    elif response.status_code == 404:
        print(response.text)
    elif response.status_code != 200:
        raise Exception(f'{response.status_code}: {response.text}')
    return response

def get_paper_data(paper_id):
    try:
        response = throttled_request(
            'GET',
            f'{SEMANTIC_SCHOLAR_API_URL}paper/{paper_id}',
            params={'fields': 'title,abstract,references.title,references.abstract'},
            headers=headers
        )
        data = response.json()
        referenced_abstracts = [ref["abstract"] for ref in data.get("references", [])]
        return {
            "paperId": paper_id,
            "title": data["title"],
            "abstract": data["abstract"],
            "referenced_abstracts": referenced_abstracts
        }
    except (KeyError, requests.exceptions.RequestException):
        return None

def most_cited(papers: List[dict], citation_count_threshold: int) -> List[str]:
    sorted_papers = sorted(papers, key=lambda x: x['citationCount'] or 0, reverse=True)
    filtered_papers = filter(lambda x: (x['citationCount'] or 0) >= citation_count_threshold, sorted_papers)
    return [x['paperId'] for x in filtered_papers]

def get_most_cited_references(paper_id: str, citation_count_threshold: int) -> List[str]:
    response = throttled_request(
        'GET',
        f'{SEMANTIC_SCHOLAR_API_URL}paper/{paper_id}',
        params={'fields': 'references.citationCount'},
    )
    paper = response.json()
    return most_cited(paper.get('references', []), citation_count_threshold)

def get_more_seed_papers(seed_paper_id: str, min_citation_count: int) -> List[str]:
    seed_paper_ids = get_most_cited_references(seed_paper_id, min_citation_count)
    for paper_id in copy.copy(seed_paper_ids):
        seed_paper_ids += get_most_cited_references(paper_id, min_citation_count)
    return seed_paper_ids

def get_citing_papers(paper_id: str, max_offset: int = 10000, limit: int = 1000) -> List[str]:
    paper_ids = []
    offset = 0
    while True:
        response = throttled_request(
            'GET',
            f'{SEMANTIC_SCHOLAR_API_URL}paper/{paper_id}/citations',
            params={'fields': 'paperId', 'offset': offset, 'limit': limit},
        )
        data = response.json()
        paper_ids += [paper['citingPaper']['paperId'] for paper in data['data']]
        offset = data['next']
        if offset >= max_offset or offset + limit >= 10000:
            break
    return paper_ids


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def main():
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/test', exist_ok=True)

    seed_paper_id = '204e3073870fae3d05bcbc2f6a8e263d9b72e776'
    seed_paper_ids = get_more_seed_papers(seed_paper_id, min_citation_count=10000)

    paper_ids = []
    for seed_paper in tqdm(seed_paper_ids, desc='Collecting citing paper ids'):
        paper_ids += get_citing_papers(seed_paper, max_offset=500, limit=100)

    all_paper_ids = list(set([seed_paper_id] + seed_paper_ids + paper_ids))
    chunk_size = 100
    
    extracted_data = []  # Initialize the extracted_data list
    
    for paper_ids_chunk in tqdm(chunker(all_paper_ids, chunk_size), desc='Collecting instance papers', total=math.ceil(len(all_paper_ids)/chunk_size)):
        response = throttled_request(
            'POST',
            f'{SEMANTIC_SCHOLAR_API_URL}paper/batch',
            params={'fields': 'title,abstract,references.title,references.abstract'},
            json={'ids': paper_ids_chunk},
            headers=headers
        )
        papers = response.json()

        # Parallelize API calls using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            paper_data = list(executor.map(get_paper_data, paper_ids_chunk))
        paper_data = [d for d in paper_data if d is not None]
        
        extracted_data.extend(paper_data)

        for paper in papers:
            if paper is not None and (paper_id := paper.get('paperId')):
                train_or_test = random.choices(['train', 'test'], weights=[0.8, 0.2])[0]
                with open(f'dataset/{train_or_test}/{paper_id}.json', 'w') as fp:
                    json.dump(paper, fp, indent=2)


if __name__ == '__main__':
    main()