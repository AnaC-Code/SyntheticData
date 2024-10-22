import pandas as pd
import copy
import json
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import lil_matrix
import numpy as np

class Data:
    def __init__(self, data:dict, metadata: dict):
        """
        Initializes the Data class with a dictionary of dataframes.
        
        :param metadata: Dictionary where keys are table names (strings) and values are metadata.
        :param data: Dictionary where keys are table names (strings) and values are dataframes.
        
        """
        self.data = data
        self.metadata = metadata

    def extend_tables(self) -> None:
        for table, columns in self.metadata["tables"].items():
            data_table=self.data[table]
            parents_dict=self.get_parents(table=table)
            children_dict=self.get_children(table=table)
            if len(parents_dict['realtionship'])>0:
                data_table = self.extend_parents(parents=parents_dict['realtionship'],data_table=data_table)
            if len(children_dict['realtionship'])>0:
                data_table = self.extend_children(children=children_dict['realtionship'],data_table=data_table)
            data_table.to_csv(f"Extended/{table}.csv",index=False)

    def extend_metadata(self) -> None:
        for table, columns in self.metadata["tables"].items():
            metadata_table = copy.deepcopy(self.metadata)
            metadata_table = metadata_table["tables"][table]
            parents_dict=self.get_parents(table=table)
            children_dict=self.get_children(table=table)
            if len(children_dict['realtionship'])>0:
                metadata_table = self.extend_metadata_children(children=children_dict['realtionship'],metadata_table=metadata_table)
            if len(parents_dict['realtionship'])>0:
                metadata_table = self.extend_metadata_parents(parents=parents_dict['realtionship'],metadata_table=metadata_table)
            self.convert_to_datatype_synthpop(metadata_table=metadata_table,table=table)

    def convert_to_datatype_synthpop(self, metadata_table=None, table=None ) -> None:
        synthpop_dict = {}
        for column,value in metadata_table["columns"].items():
            if value["sdtype"] == "categorical":
                synthpop_dict[column] = "category"
            if value["sdtype"] == "numerical":
                if value["computer_representation"] == "Int64" or value["computer_representation"] == "Int32":
                    synthpop_dict[column] = "int"
                if value["computer_representation"] == "Float":
                    synthpop_dict[column] = "float"

        with open(f'Metadata/{table}.json', 'w') as json_file:
            json.dump(synthpop_dict, json_file, indent=4)

    def extend_parents(self,parents=None,data_table=None) -> None:
        for parent in parents:
            parent_table = parent["parent_table_name"]
            data_parent = self.data[parent_table]
            data_parent_copy=data_parent.copy()
            name_child_foreign_key=parent["child_foreign_key"]
            name_parent_primary_key=parent["parent_primary_key"]
            data_parent_prefixed = data_parent_copy.add_prefix(f"{parent_table}_{name_child_foreign_key}_")
            data_table = data_table.merge(data_parent_prefixed, left_on=parent["child_foreign_key"], right_on=f'{parent_table}_{name_child_foreign_key}_{name_parent_primary_key}', how='left')
        return data_table
    
    def extend_children(self,children=None,data_table=None) -> None:
        for child in children:
            child_table = child["child_table_name"]
            data_child = self.data[child_table]
            children_counts = data_child[child["child_foreign_key"]].value_counts().reset_index()
            name_foreing_key = child["child_foreign_key"]
            children_counts.columns = [child["parent_primary_key"], f'count_{child_table}_{name_foreing_key}']
            data_table = data_table.merge(children_counts, on=child["parent_primary_key"], how='left')
            count_column_name = f'count_{child_table}_{name_foreing_key}'
            #If the parent table doesnt have any reference to the child table, then the number of references is 0
            data_table[count_column_name].fillna(0, inplace=True)
        return data_table

    def get_children(self,table=None) -> None:
        result={}
        result["realtionship"]=[]
        for realtionship in self.metadata["relationships"]:
            if realtionship["parent_table_name"] == table:
                result["realtionship"].append(realtionship)
        return result

    def get_parents(self,table=None) -> None:
        result={}
        result["realtionship"]=[]
        for realtionship in self.metadata["relationships"]:
            if realtionship["child_table_name"] == table:
                result["realtionship"].append(realtionship)
        return result

    def add_prefix_to_keys(self,d, prefix):
        return {prefix + key: value for key, value in d.items()}

    def extend_metadata_parents(self,parents=None,metadata_table=None) -> None:
        for parent in parents:
            parent_table = parent["parent_table_name"]
            child_foreign_key = parent["child_foreign_key"]
            metadata_parent = self.metadata["tables"][parent_table]["columns"]
            metadata_parent_copy=self.add_prefix_to_keys(metadata_parent, f'{parent_table}_{child_foreign_key}_')
            metadata_table["columns"].update(metadata_parent_copy)
        return metadata_table
    
    def extend_metadata_children(self,children=None,metadata_table=None) -> None:
        for child in children:
            child_table = child["child_table_name"]
            name_foreing_key = child["child_foreign_key"]
            #We need to create a new column for the number of references
            metadata_table["columns"][f'count_{child_table}_{name_foreing_key}'] = {
                "sdtype": "numerical",
                "computer_representation": "Int32"
            }
        return metadata_table

    def connect_parents(self) -> None:
        for table, columns in self.metadata["tables"].items():
            fake_table= pd.read_csv(f"Synthetic/synthpop_fake_{table}.csv")
            primary_key = columns.get("primary_key")
            fake_table[primary_key]= range(len(fake_table))
            fake_table.to_csv(f"Synthetic_Data/fake_{table}.csv",index=False)

        for table, columns in self.metadata["tables"].items():
            parents_dict=self.get_parents(table=table)
            if len(parents_dict['realtionship'])>0:
                self.link_parents(parents=parents_dict['realtionship'],table_name=table)

    def link_parents(self,parents=None,table_name=None) -> None:
        for parent in parents:
            parent_table = parent["parent_table_name"]
            self.linked(parent_name=parent_table,table_name=table_name,foreing_key=parent["child_foreign_key"],parent_key=parent["parent_primary_key"])

    def linked(self,parent_name=None,table_name=None,foreing_key=None,parent_key=None) -> None:
        data_copy = copy.deepcopy(self.metadata["tables"][parent_name]["columns"])
        pk_parent_table=self.metadata["tables"][parent_name]["primary_key"]
        pk_child_table=self.metadata["tables"][table_name]["primary_key"]
        filtered_data_copy = {key: value for key, value in data_copy.items() if value.get('sdtype') != 'id'}
        keys_list_parent = list(filtered_data_copy.keys())
        prefix_parent_name_foreing_key = f'{parent_name}_{foreing_key}_'
        key_columns_parent =  [prefix_parent_name_foreing_key+ s for s in keys_list_parent]
        parent_table= pd.read_csv(f"Synthetic/synthpop_fake_{parent_name}.csv")
        parent_table[parent_key] = range(len(parent_table))
        parent_table.to_csv(f"Synthetic/synthpop_fake_{parent_name}.csv", index=False)
        parent_table =parent_table[keys_list_parent]
        child_table= pd.read_csv(f"Synthetic/synthpop_fake_{table_name}.csv")
        child_table[pk_child_table] = range(len(child_table))
        child_table =child_table[key_columns_parent]
        return self.transformed_dataframe(parent_table=parent_table, child_table=child_table,parent_table_name=parent_name,child_table_name=table_name,pk_parent_table=pk_parent_table,pk_child_table=pk_child_table,foreing_key=foreing_key )

    def transform_categorical_columns(self,
                                      categorical_columns_parent=None, 
                                      parent_table_name=None,
                                      parent_table=None,
                                      child_table=None ):
        encoder = OneHotEncoder(sparse_output=False)

        if len(categorical_columns_parent)>0:
            parent_encoded = encoder.fit_transform(self.data[parent_table_name][categorical_columns_parent])
            parent_encoded = encoder.transform(parent_table[categorical_columns_parent])
            child_encoded = encoder.transform(child_table[categorical_columns_parent])
            try:
                feature_names = encoder.get_feature_names_out(categorical_columns_parent)
            except AttributeError:
                feature_names = encoder.get_feature_names(categorical_columns_parent)
            child_encoded_df = pd.DataFrame(child_encoded, columns=feature_names)
            parent_encoded_df = pd.DataFrame(parent_encoded, columns=feature_names)
            return child_encoded_df,parent_encoded_df
        else:
            # If no categorical columns, create empty dataframes
            return pd.DataFrame(), pd.DataFrame()

    def transform_dataframe_to_array(self,
                                    parent_table=None,
                                    numerical_columns_parent=None,
                                    child_table=None,
                                    parent_table_name=None,
                                    parent_encoded_df=None,
                                    child_encoded_df=None):

        parent_numerical_df = parent_table[numerical_columns_parent]
        parent_numerical_df.reset_index(drop=True, inplace=True)
        parent_encoded_df.reset_index(drop=True, inplace=True)
        parent_encoded_df = pd.concat([parent_encoded_df, parent_numerical_df], axis=1)
        child_numerical_df = child_table[numerical_columns_parent]
        child_numerical_df.reset_index(drop=True, inplace=True)
        child_encoded_df.reset_index(drop=True, inplace=True)
        child_encoded_df = pd.concat([child_encoded_df, child_numerical_df], axis=1)
 
        self.normalize_columns(table=child_encoded_df,parent_table_name=parent_table_name,numerical_columns=numerical_columns_parent)
        self.normalize_columns(table=parent_encoded_df,parent_table_name=parent_table_name,numerical_columns=numerical_columns_parent)
        child_array = child_encoded_df.values
        parent_array = parent_encoded_df.values
        return child_array,parent_array
    
    def connect_closest_child(self,
                                    pk_child_table=None,
                                    child_table=None,
                                    parent_table=None,
                                    pk_parent_table=None,
                                    pk_parent_column=None,
                                    child_table_name=None,
                                    foreing_key=None,
                                    count_parent_column=None,
                                    parent_array=None,
                                    child_array=None,
                                    similarity_array=None):

        child_table[pk_child_table] = range(len(child_table))
        parent_table[pk_parent_table]=  pk_parent_column.reset_index(drop=True)
        parent_table[f'count_{child_table_name}_{foreing_key}']=  count_parent_column.reset_index(drop=True)
        child_table[f'{foreing_key}']=None
        assignments = pd.DataFrame(index=child_table.index, columns=[pk_child_table, foreing_key])
        assignments[pk_child_table] = child_table[pk_child_table]
        parent_assignment_counts = {parentid: 0 for parentid in parent_table[pk_parent_table]}
        max_assignments = {row[pk_parent_table]: row[f'count_{child_table_name}_{foreing_key}'] for index, row in parent_table.iterrows()}
        n, m = parent_array.shape[0], child_array.shape[0]
        for child_idx in range(m):
            sorted_indices = np.argsort(similarity_array[child_idx])
            for parent_idx in sorted_indices:
                parentid = parent_table.loc[parent_idx, pk_parent_table]
                if parent_assignment_counts[parentid] < max_assignments[parentid]:
                    assignments.at[child_idx, foreing_key] = parentid
                    parent_assignment_counts[parentid] += 1
                    break
        return assignments
    
    def update_final_dataframes(self,
                            assignments=None,
                            child_table=None,
                            foreing_key=None,
                            child_table_name=None,
                            parent_table_name=None,
                            pk_child_table=None,
                            count_parent_column=None,
                            parent_table=None,
                            pk_parent_table=None):

        child_table[foreing_key] = assignments[foreing_key]
        final_table = pd.read_csv(f"Synthetic_Data/fake_{child_table_name}.csv")
        final_table[foreing_key]=child_table[foreing_key]
        prefix = f'{parent_table_name}_{foreing_key}_'
        cols_to_drop = [col for col in final_table.columns if col.startswith(prefix)]
        final_table = final_table.drop(columns=cols_to_drop)
        final_table[pk_child_table]=range(len(child_table))
        final_table.to_csv(f"Synthetic_Data/fake_{child_table_name}.csv",index=False)
        parent_table[f'count_{child_table_name}_{foreing_key}']=  count_parent_column.reset_index(drop=True)
        columns_count_to_delete = []
        columns_count_to_delete.append(f'count_{child_table_name}_{foreing_key}')
        parent_table_copy_final=pd.read_csv(f"Synthetic_Data/fake_{parent_table_name}.csv")
        parent_table_copy_final.drop(columns=columns_count_to_delete,inplace=True)
        parent_table_copy_final[pk_parent_table]= range(len(parent_table_copy_final))
        parent_table_copy_final.to_csv(f"Synthetic_Data/fake_{parent_table_name}.csv",index=False)
    
        
    def process_datasets(self,
                        parent_table=None,
                        numerical_columns_parent=None,
                        child_table=None,
                        parent_table_name=None,
                        pk_child_table=None,
                        pk_parent_table=None,
                        pk_parent_column=None,
                        child_table_name=None,
                        foreing_key=None,
                        count_parent_column=None,
                        categorical_columns_parent=None):

        child_encoded_df,parent_encoded_df = self.transform_categorical_columns(
                                                categorical_columns_parent=categorical_columns_parent, 
                                                parent_table_name=parent_table_name,
                                                parent_table=parent_table,
                                                child_table=child_table)

        child_array,parent_array = self.transform_dataframe_to_array(
                                        parent_table=parent_table,
                                        numerical_columns_parent=numerical_columns_parent,
                                        child_table=child_table,
                                        parent_table_name=parent_table_name,
                                        parent_encoded_df=parent_encoded_df,
                                        child_encoded_df=child_encoded_df)

        similarity_array = self.similarity_matrix(child_array=child_array,parent_array=parent_array)

        assignments = self.connect_closest_child(
                                                pk_child_table=pk_child_table,
                                                child_table=child_table,
                                                parent_table=parent_table,
                                                pk_parent_table=pk_parent_table,
                                                pk_parent_column=pk_parent_column,
                                                child_table_name=child_table_name,
                                                foreing_key=foreing_key,
                                                count_parent_column=count_parent_column,
                                                parent_array=parent_array,
                                                child_array=child_array,
                                                similarity_array=similarity_array)

        self.update_final_dataframes(assignments=assignments,
                                    child_table=child_table,
                                    foreing_key=foreing_key,
                                    child_table_name=child_table_name,
                                    parent_table_name=parent_table_name,
                                    pk_child_table=pk_child_table,
                                    count_parent_column=count_parent_column,
                                    parent_table=parent_table,
                                    pk_parent_table=pk_parent_table)


    def transformed_dataframe(self,parent_table=None, child_table=None,parent_table_name=None,child_table_name=None,pk_parent_table=None,pk_child_table=None,foreing_key=None):
        parent_data_copy = copy.deepcopy(self.metadata["tables"][parent_table_name]["columns"])
        categorical_columns_parent=[]
        numerical_columns_parent=[]
        parent_table_from_synthpop= pd.read_csv(f"Synthetic/synthpop_fake_{parent_table_name}.csv")
        parent_table[f'count_{child_table_name}_{foreing_key}'] = parent_table_from_synthpop[f'count_{child_table_name}_{foreing_key}']
        parent_table[pk_parent_table] = parent_table_from_synthpop[pk_parent_table]
        parent_table = parent_table[parent_table[f'count_{child_table_name}_{foreing_key}'] > 0]
        parent_table.reset_index(drop=True, inplace=True)
        pk_parent_column = parent_table[pk_parent_table]
        count_parent_column = parent_table[f'count_{child_table_name}_{foreing_key}']

        for key, value in parent_data_copy.items():
            if value.get('sdtype') == 'categorical':
                categorical_columns_parent.append(key)
            if value.get('sdtype') == 'numerical':
                numerical_columns_parent.append(key)

        child_table.columns = child_table.columns.str.replace(f'{parent_table_name}_{foreing_key}_', '', regex=True)
        self.process_datasets(parent_table=parent_table,
                                numerical_columns_parent=numerical_columns_parent,
                                child_table=child_table,
                                parent_table_name=parent_table_name,
                                pk_child_table=pk_child_table,
                                pk_parent_table=pk_parent_table,
                                pk_parent_column=pk_parent_column,
                                child_table_name=child_table_name,
                                foreing_key=foreing_key,
                                count_parent_column=count_parent_column,
                                categorical_columns_parent=categorical_columns_parent)

    def similarity_matrix(self, child_array=None, parent_array=None):
        if child_array is None or parent_array is None:
            raise ValueError("Both child_array and parent_array must be provided")
        
        n, m = parent_array.shape[0], child_array.shape[0]
        similarity_matrix = lil_matrix((m, n), dtype=np.float32)

        # Define smaller chunk size
        chunk_size = 100  # Adjust chunk size to a smaller value to fit memory constraints

        # Ensure arrays are in float32 to save memory
        parent_array = parent_array.astype(np.float32)
        child_array = child_array.astype(np.float32)

        # Compute distances in chunks
        for i in range(0, m, chunk_size):
            for j in range(0, n, chunk_size):
                # Compute the end of the chunk
                i_end = min(i + chunk_size, m)
                j_end = min(j + chunk_size, n)
                
                # Extract the chunks
                movie_chunk = child_array[i:i_end]
                actors_chunk = parent_array[j:j_end]
                
                # Compute the squared differences
                diff = actors_chunk[:, np.newaxis, :] - movie_chunk[np.newaxis, :, :]
                squared_diff = diff ** 2
                
                # Compute the Euclidean distances
                distances = np.sqrt(np.sum(squared_diff, axis=2))
                
                # Store the results in the sparse matrix
                similarity_matrix[i:i_end, j:j_end] = distances.T

        similarity_array = similarity_matrix.toarray()
        return similarity_array

    def normalize_columns(self,table=None,parent_table_name=None,numerical_columns=None):
        for column in numerical_columns:
            max = self.data[parent_table_name][column].max()
            min = self.data[parent_table_name][column].min()
            self.normalize_dataframe(df=table,max=max,min=min,columnName=column)

    def normalize_dataframe(self,df=None,max=None,min=None,columnName=None):
        df[columnName] = (df[columnName] - min) / (max - min)
