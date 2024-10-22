from sdv.datasets.demo import download_demo
from Data.Data import Data  # Import the Data class from data.py 
import os
import shutil

def delete_html_and_png_files(folder_path=None):
    if folder_path is None:
        folder_path = os.getcwd()

    # List all items in the folder
    for item in os.listdir(folder_path):
        # Construct the full item path
        item_path = os.path.join(folder_path, item)
        
        # Check if it's a file (not a directory) and if it ends with .html or .png
        if os.path.isfile(item_path) and (item.endswith('.html') or item.endswith('.png')):
            # Delete the file
            os.remove(item_path)
            print(f"Deleted file: {item_path}")

def check_for_nans(dataframes_dict, metadata):
    for key, df in dataframes_dict.items():
        if df.isna().any().any():
            return False

    for key, value in metadata["tables"].items():
        if 'primary_key' not in value:
            return False

    return True

def clear_folder(folder_path):
    """
    Deletes all contents of the specified folder without deleting the folder itself.

    Parameters:
    folder_path (str): The path to the folder whose contents will be deleted.
    """
    # Ensure the provided path is a directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return  # The folder was just created, so it is empty.

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove the file or link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove the directory
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')

    print(f"All contents of the folder '{folder_path}' have been deleted.")

delete_html_and_png_files()
clear_folder('Synthetic_Data')
clear_folder('Metadata')
clear_folder('Extended')
clear_folder('Synthetic')

while True:
    with open('dataset_input.txt', 'r') as file:
        user_input = file.read().strip()
    data, metadata = download_demo(
        modality = 'multi_table',
        dataset_name = str(user_input)
    )
    if check_for_nans(data, metadata.to_dict()):
        data_manager = Data(data=data, metadata=metadata.to_dict())
        data_manager.extend_tables()
        data_manager.extend_metadata()
        break
    else:
        print("The data contain null values, please select another dataset.")
 