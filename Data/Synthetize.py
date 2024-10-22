from synthpop import Synthpop
import pandas as pd
from pathlib import Path
import json
import os
import math

class Synthetize:

    def start_variables(self) -> None:
        folder_path = 'Extended'
        files = os.listdir(folder_path)
        csv_files = [file for file in files if file.endswith('.csv')]
        for csv_file in csv_files:
            file_name = os.path.splitext(csv_file)[0]
            pre_data = pd.read_csv(f'Extended/{file_name}.csv')
            dtypes_path = Path(f"Metadata/{file_name}.json")
            dtypes={}
            total_records = len(pre_data)
            ten_percent = math.ceil(total_records * 0.10)
            with dtypes_path.open('r') as f:
                dtypes = json.load(f)
            processed_table = self.process_table(table=pre_data, dtypes=dtypes) 
            self.synthethize_table(table=processed_table, dtype=dtypes, sample=ten_percent,table_Name=file_name)

    def process_table(self, table=None, dtypes=None) -> None:
        pre_data = table[[col for col in dtypes.keys() if col in table.columns]]
        for col, dtype in dtypes.items():
            if dtype == 'int':
                if col.startswith("count_"):
                    pre_data[col] = pre_data[col].fillna(0)
                else:
                    min_value = pre_data[col].min(skipna=True)    
                    pre_data[col] = pre_data[col].fillna(min_value-1)
                pre_data[col] = pre_data[col].astype(int)
            elif dtype == 'float':
                min_value = pre_data[col].min(skipna=True)
                pre_data[col] = pre_data[col].fillna(min_value-1)
                pre_data[col] = pre_data[col].astype(float)
            elif dtype == 'datetime':
                pre_data[col] = pd.to_datetime(pre_data[col])
            elif dtype == 'category':
                pre_data[col] = pre_data[col].astype('category')
            elif dtype == 'bool':
                pre_data[col] = pre_data[col].astype(bool)     
        return pre_data

    def synthethize_table(self, table=None, dtype=None, sample=None,table_Name=None) -> None:
        spop = Synthpop()
        spop.fit(table, dtype)
        synth_df = spop.generate(sample)
        synth_df.to_csv(f"Synthetic/synthpop_fake_{table_Name}.csv",index=False)


 