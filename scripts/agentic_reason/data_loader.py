import os
import json

def load_dataset(dataset_name, split, data_dir='./data'):
    # Adjust parameters based on dataset
    if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        MAX_SEARCH_LIMIT = 5
        if dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki']:
            MAX_SEARCH_LIMIT = 10
            MAX_TURN = 15
        top_k = 10
        max_doc_len = 3000

    # Data paths based on dataset
    if dataset_name == 'livecode':
        data_path = f'{data_dir}/LiveCodeBench/{split}.json'
    elif dataset_name in ['math500', 'gpqa', 'aime', 'amc']:
        data_path = f'{data_dir}/{dataset_name.upper()}/{split}.json'
    else:
        data_path = f'{data_dir}/QA_Datasets/{dataset_name}.json'

    # Create data directories if they don't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Create sample data file if it doesn't exist
    if not os.path.exists(data_path):
        create_sample_data(dataset_name, data_path)

    # Check if data file exists and is not empty
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please ensure you have downloaded the dataset and placed it in the correct location:\n"
            f"For {dataset_name.upper()}: {data_path}"
        )

    print('-----------------------')
    print(f'Using {dataset_name} {split} set from: {data_path}')
    print('-----------------------')

    with open(data_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

def create_sample_data(dataset_name, data_path):
    """Create sample data for testing purposes when actual dataset is not available"""
    sample_data = []
    
    if dataset_name == 'gpqa':
        sample_data = [{
            "Question": "What is the capital of France?",
            "Choices": ["London", "Paris", "Berlin", "Madrid"],
            "Answer": "Paris",
            "input_output": json.dumps({
                "inputs": ["What is the capital of France?"],
                "outputs": ["Paris"]
            })
        }]
    elif dataset_name in ['math500', 'aime', 'amc']:
        sample_data = [{
            "Question": "What is 2 + 2?",
            "Answer": "4",
            "input_output": json.dumps({
                "inputs": ["What is 2 + 2?"],
                "outputs": ["4"]
            })
        }]
    elif dataset_name == 'livecode':
        sample_data = [{
            "Question": "Write a function to add two numbers",
            "question_title": "Add Two Numbers",
            "input_output": json.dumps({
                "inputs": ["def add(a, b):\n    return a + b"],
                "outputs": ["def add(a, b):\n    return a + b"]
            })
        }]
    
    print(f"Creating sample data file at: {data_path}")
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    print("Sample data file created for testing purposes.")