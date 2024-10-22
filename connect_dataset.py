from sdv.datasets.demo import download_demo
import pandas as pd
from Data.Data import Data  # Import the Data class from data.py

df = pd.DataFrame()
with open('dataset_input.txt', 'r') as file:
    user_input = file.read().strip()

data, metadata = download_demo(
    modality='multi_table',
    dataset_name=str(user_input)
)

data_manager = Data(data=data, metadata=metadata.to_dict())
data_manager.connect_parents()
  