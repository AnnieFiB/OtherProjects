import os
import sys
import kaggle
import pandas as pd
import requests
from zipfile import ZipFile
from io import BytesIO

def fetch_kaggle_dataset(search_query="human resources"):
    """
    Authenticate Kaggle API, search for datasets, download, list available files, and allow user input to select a dataset.

    Parameters:
    search_query (str): The keyword for searching datasets on Kaggle.

    Returns:
    pd.DataFrame: Loaded dataset as a Pandas DataFrame.
    """
    # Dynamically resolve the Kaggle config directory
    venv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.venv'))
    kaggle_config_dir = os.path.join(venv_path, '.kaggle')

    # Set the KAGGLE_CONFIG_DIR environment variable
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_config_dir

    # Authenticate with Kaggle API
    kaggle.api.authenticate()

    # Search for datasets related to the query
    search_result = kaggle.api.dataset_list(search=search_query)

    if not search_result:
        print("❌ No datasets found for the search query.")
        return None

    # Limit results to the top 5 datasets
    top_datasets = search_result[:5]

    # Print dataset details and list available files
    print("\n🔹 Available Datasets:")
    dataset_refs = []
    file_info_dict = {}
    
    for i, dataset in enumerate(top_datasets):
        print(f"\nDataset {i + 1}: {dataset.ref} - {dataset.title}")
        dataset_refs.append(dataset.ref)

        # Download the dataset ZIP file into memory
        api_url = f'https://www.kaggle.com/api/v1/datasets/download/{dataset.ref}'
        response = requests.get(api_url, stream=True)
        zip_file = ZipFile(BytesIO(response.content))

        # List all files in the ZIP archive
        print("Files:")
        file_list = []
        for file_info in zip_file.infolist():
            file_name = file_info.filename
            file_size = file_info.file_size
            print(f"  - {file_name} (Size: {file_size} bytes)")
            file_list.append((file_name, file_size, file_info))
        
        # Store file info for the dataset
        file_info_dict[i + 1] = (dataset.ref, file_list)

    # Allow user to select dataset by number
    while True:
        try:
            dataset_index = int(input("\nEnter the number of the dataset you want to use: "))
            
            # Validate dataset index
            if dataset_index < 1 or dataset_index > len(dataset_refs):
                print(f"❌ Invalid dataset number. Please enter a number between 1 and {len(dataset_refs)}.")
                continue
            
            break  # Exit loop if input is valid
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")

    # Get the selected dataset and its files
    selected_dataset_ref, file_list = file_info_dict[dataset_index]

    # If there are multiple files, let the user choose one
    if len(file_list) > 1:
        print("\n🔹 Available Files:")
        for j, (file_name, file_size, _) in enumerate(file_list):
            print(f"{j + 1}. {file_name} (Size: {file_size} bytes)")
        
        while True:
            try:
                file_index = int(input("\nEnter the number of the file you want to use: "))
                
                # Validate file index
                if file_index < 1 or file_index > len(file_list):
                    print(f"❌ Invalid file number. Please enter a number between 1 and {len(file_list)}.")
                    continue
                
                break  # Exit loop if input is valid
            except ValueError:
                print("❌ Invalid input. Please enter a valid number.")
        
        selected_file_name, _, selected_file_info = file_list[file_index - 1]
    else:
        # If there's only one file, select it automatically
        selected_file_name, _, selected_file_info = file_list[0]

    # Download the selected dataset ZIP file into memory
    api_url = f'https://www.kaggle.com/api/v1/datasets/download/{selected_dataset_ref}'
    response = requests.get(api_url, stream=True)
    zip_file = ZipFile(BytesIO(response.content))

    # Open the selected file
    file_ext = os.path.splitext(selected_file_name)[1].lower()
    file = zip_file.open(selected_file_name)

    # Load the file based on its type
    if file_ext == ".csv":
        data = pd.read_csv(file)
    elif file_ext == ".xlsx":
        data = pd.read_excel(file)
    elif file_ext == ".json":
        data = pd.read_json(file)
    else:
        raise ValueError(f"❌ Unsupported file type: {file_ext}")

    print("\n✅ Dataset loaded successfully!")
    return data