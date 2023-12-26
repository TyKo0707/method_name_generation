import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.code_generation import format_text_for_code_gen
from environs import Env

env = Env()
env.read_env()
UNPROCESSED_FUNCTIONS_DATASET = env.str("UNPROCESSED_FUNCTIONS_DATASET")
MODEL_CHECKPOINT = env.str("MODEL_CHECKPOINT")
MAIN_DATASET = env.str("MAIN_DATASET")


def extract_input_output_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts input and output data for code generation from a DataFrame and constructs a new DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing function information.

    Returns:
    - pd.DataFrame: A new DataFrame containing columns for function_id, input, label, is_multiline, is_test, and is_abstract.
    """
    inputs = []

    for index in tqdm(range(df.shape[0])):
        row = df.iloc[index]
        row_input = format_text_for_code_gen(row)
        inputs.append(row_input)

    return pd.DataFrame({'function_id': df.function_id, 'input': inputs, 'label': df.identifier,
                         'is_multiline': df.is_multiline, 'is_test': df.is_test, 'is_abstract': df.is_abstract})


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    functions_df = pd.read_csv(UNPROCESSED_FUNCTIONS_DATASET, index_col=0).sample(frac=1).reset_index(drop=True)
    processed_io_df = extract_input_output_from_df(functions_df[:200000])
    processed_io_df = processed_io_df.dropna(subset=['input', 'label'])
    processed_io_df.to_parquet(MAIN_DATASET, engine='fastparquet', compression='gzip', index=False)
