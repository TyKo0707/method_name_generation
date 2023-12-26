import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.code_generation import format_text_for_code_gen


def extract_input_output_from_df(df):
    inputs = []

    for index in tqdm(range(df.shape[0])):
        row = df.iloc[index]
        row_input = format_text_for_code_gen(row)
        inputs.append(row_input)

    return pd.DataFrame({'function_id': df.function_id, 'input': inputs, 'label': df.identifier,
                         'is_multiline': df.is_multiline, 'is_test': df.is_test, 'is_abstract': df.is_abstract})


if __name__ == '__main__':
    checkpoint = "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    functions_df = pd.read_csv('datasets/functions_df.csv', index_col=0).sample(frac=1).reset_index(drop=True)
    processed_io_df = extract_input_output_from_df(functions_df)
    processed_io_df = processed_io_df.dropna(subset=['input', 'label'])
    processed_io_df.to_parquet('datasets/functions_df_inputs_outputs.parquet.gz',
                               engine='fastparquet', compression='gzip', index=False)

    # from datasets import Dataset, DatasetDict
    # max_input_length = 512
    # max_target_length = 32
    # def preprocess_function(dataframe):
    #     model_inputs = {'input_ids': [], 'attention_mask': [], 'label': []}
    #     for index in tqdm(range(dataframe.shape[0])):
    #         row = dataframe.iloc[index]
    #         input_ids = tokenizer(row.input, truncation=True, max_length=max_input_length, padding="max_length")
    #         label = tokenizer(row.label, truncation=True, max_length=max_target_length, padding="max_length")
    #         model_inputs['input_ids'].append(input_ids['input_ids'])
    #         model_inputs['attention_mask'].append(input_ids['attention_mask'])
    #         model_inputs['label'].append(label['input_ids'])
    #     return model_inputs
    #
    #
    # processed_io_df_wo_abstract = processed_io_df[(processed_io_df.is_abstract == False)].reset_index(drop=True)
    # model_inputs = preprocess_function(processed_io_df_wo_abstract)
    # dataset = Dataset.from_dict(model_inputs)
    #
    # # 80% train, 10% test + 10% validation
    # dataset_train_devtest = dataset.train_test_split(test_size=0.2, seed=42)
    # dataset_devtest = dataset_train_devtest['test'].train_test_split(test_size=0.5, seed=42)
    # dataset_splits = DatasetDict({
    #     'train': dataset_train_devtest['train'],
    #     'valid': dataset_devtest['train'],
    #     'test': dataset_devtest['test']
    # })
    #
    # dataset_file_name = 'datasets/functions_dataset_splits.hf'
    # dataset_splits.save_to_disk(dataset_file_name)
