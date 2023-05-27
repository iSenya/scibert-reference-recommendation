# scibert-reference-recommendation

This repository contains the code for a recommendation engine for scientific paper citations based on a written text (abstract). The project involves retrieving papers from the Semantic Scholar API, preprocessing the text data, training a model using BERT-based embeddings, and evaluating the model's performance.

## Project Structure

The project consists of the following files:

- `semanticscholar_api_call.py`: Creates the dataset(s) for training by making API calls to Semantic Scholar and extracting relevant information such as paper ID, title, abstract, and references.

- `embedding_by_label.py`: Defines the model class, trainer, and processes the dataset for training. This file also trains the model using the processed dataset.

- `preprocessing.py`: Preprocesses the natural text data by removing links and entities from the text.

- `ModelForEval.py`: Creates the class with the architecture of the network without the classification head. This class is used for generating embedding vectors.

- `random_api_call.py`: Retrieves random papers from Semantic Scholar, adds the references of the initial paper, and creates a file with paper ID, title, and the embedding vector created from the abstract.

- `model_evaluation.py`: Compares the embedding vector of a given abstract with each of the embedding vectors constructed in the previous step, and evaluates the performance of the recommendation engine.

## Usage

1. Run `semanticscholar_api_call.py` to create the dataset(s) for training. Adjust the API call parameters as needed.

2. Execute `embedding_by_label.py` to process the dataset and train the model. Modify the hyperparameters and training configurations as desired.

3. Utilize `random_api_call.py` to retrieve random papers, add references, and generate embedding vectors. Specify the necessary API parameters and settings.

4. Finally, run `model_evaluation.py` to evaluate the performance of the recommendation engine using the embedding vectors. Adjust the evaluation process and parameters as required.

Note: Ensure that the required dependencies are installed by referring to the `requirements.txt` file.

## License

This project is licensed under the [MIT License](LICENSE).

