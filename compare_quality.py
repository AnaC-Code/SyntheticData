import pandas as pd
from sdv.datasets.demo import download_demo
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import run_diagnostic
from sdv.evaluation.multi_table import evaluate_quality

def get_synthethic_data(metadata,name):
    fake_dictionary = {}    
    for key, value in metadata["tables"].items():
        df = pd.read_csv(f"Synthetic_Data/fake_{key}.csv")
        fake_dictionary[key] = df
    return fake_dictionary

def set_information(real_data=None, synthetic_data=None,metadata=None):

    diagnostic = run_diagnostic(
            real_data=real_data,
            synthetic_data=synthetic_data,
            metadata=metadata
    )
    quality_report = evaluate_quality(
                                real_data,
                                synthetic_data,
                                metadata
    )
    print(diagnostic.get_score())
    print(quality_report.get_score())


with open('dataset_input.txt', 'r') as file:
    user_input = file.read().strip()
real_data, metadata = download_demo(
        modality='multi_table',
        dataset_name=str(user_input)
    )

try:
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(real_data)
    hma_synthetic_data = synthesizer.sample(scale=0.1)
except Exception as e:
    synthetic_data=None

new_method_synthethic_data = get_synthethic_data(metadata=metadata.to_dict(),name=user_input)
set_information(real_data=real_data, synthetic_data=new_method_synthethic_data,metadata=metadata)
set_information(real_data=real_data, synthetic_data=hma_synthetic_data,metadata=metadata)

