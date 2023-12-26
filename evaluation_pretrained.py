import os
import pandas as pd
from utils.code_generation import extract_tokens_from_camel
from utils.metrics import code_gen_f1_score_and_accuracy, code_gen_name_similarity
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from torch import Tensor


def evaluate_and_save_results(df: pd.DataFrame, model: T5ForConditionalGeneration, tokenizer: AutoTokenizer,
                              mean_values_file_name: str, result_file_name: str, num_samples=1000):
    """
    Evaluates a model's predictions on a DataFrame, saves the results, and computes mean values.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing function information.
    - model (T5ForConditionalGeneration): The pre-trained T5 model for code generation.
    - tokenizer (AutoTokenizer): The tokenizer for the model.
    - mean_values_file_name (str): The name of the file to save mean values of evaluation metrics.
    - result_file_name (str): The name of the file to save detailed evaluation results.
    - num_samples (int, optional): The number of samples to evaluate. Default is 1000.
    """
    ftype = mean_values_file_name.split('_')[-1]
    try:
        os.mkdir(f'datasets/{ftype}')
    except FileExistsError:
        pass
    result = predict_and_evaluate(df[:num_samples], model, tokenizer, f'datasets/{ftype}/{mean_values_file_name}.csv')
    result.to_csv(f'datasets/{ftype}/{result_file_name}.csv', index=True)
    mean_values = result[['precision', 'recall', 'f1_score', 'accuracy', 'similarity']].mean(axis=0)
    mean_values.to_csv(f'datasets/{ftype}/{mean_values_file_name}.csv', index=True)


def evaluate_output(generated_result: Tensor, reference: list[str], tokenizer: AutoTokenizer) -> tuple[tuple, str]:
    """
   Evaluates the generated result against a reference and computes evaluation metrics.

   Parameters:
   - generated_result (Tensor): The generated output tensor from the model.
   - reference (list[str]): The list of reference tokens.
   - tokenizer (AutoTokenizer): The tokenizer for the model.

   Returns:
   - tuple: A tuple containing evaluation metrics (precision, recall, f1_score, accuracy, similarity)
   and the final generated output.
   """
    decoded_result = tokenizer.decode(generated_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    try:
        final = max(decoded_result.split(), key=len)
    except:
        final = ''

    prediction = extract_tokens_from_camel(final)
    metrics = code_gen_f1_score_and_accuracy(prediction, reference)
    metrics = metrics + (code_gen_name_similarity(prediction, reference),)
    return metrics, final


def intermediate_save(df: pd.DataFrame, index: int, interval_value: int, file_name: str):
    """
    Performs intermediate saves of mean values during processing.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing evaluation results.
    - index (int): The current index in the DataFrame.
    - interval_value (int): The interval at which to perform intermediate saves.
    - file_name (str): The name of the file to save intermediate mean values.
    """
    if (index + 1) % interval_value == 0:
        result = df[:index][['precision', 'recall', 'f1_score', 'accuracy', 'similarity']].mean(axis=0)
        result.to_csv(file_name, index=True)


def predict_and_evaluate(dataset: pd.DataFrame, model: T5ForConditionalGeneration, tokenizer: AutoTokenizer,
                         mean_values_file_name: str) -> pd.DataFrame:
    """
    Predicts and evaluates outputs for a given dataset using a CodeT5+ model.

    Parameters:
    - dataset (pd.DataFrame): The input DataFrame containing function information.
    - model (T5ForConditionalGeneration): The pre-trained T5 model for code generation.
    - tokenizer (AutoTokenizer): The tokenizer for the model.
    - mean_values_file_name (str): The name of the file to save mean values of evaluation metrics.

    Returns:
    - pd.DataFrame: A DataFrame containing detailed evaluation results.
    """
    result_df = dataset.copy()

    # Initialize new columns
    new_columns = ['prediction', 'precision', 'recall', 'f1_score', 'accuracy', 'similarity']
    result_df[new_columns] = 0.0

    columns_to_remove = dataset.columns.values.tolist()
    columns_to_remove.remove('label')
    result_df.drop(columns=columns_to_remove, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    result_df['prediction'] = result_df['prediction'].astype(str)

    for index in tqdm(range(dataset.shape[0])):
        row = dataset.iloc[index]

        reference = extract_tokens_from_camel(row.label)
        text = row.input

        input_ids = tokenizer.encode(text, max_length=512, truncation=True, return_tensors="pt")

        generated_ids = model.generate(input_ids, max_length=len(reference) + 2)

        metrics, prediction = evaluate_output(generated_ids[0], reference, tokenizer)
        result_df.loc[index, ['precision', 'recall', 'f1_score', 'accuracy', 'similarity']] = metrics
        result_df.loc[index, 'prediction'] = prediction

        intermediate_save(result_df, index, 250, mean_values_file_name)

    return result_df


if __name__ == '__main__':
    from utils.process_dataframe import split_into_types
    from environs import Env

    env = Env()
    env.read_env()
    MODEL_CHECKPOINT = env.str("MODEL_CHECKPOINT")
    MAIN_DATASET = env.str("MAIN_DATASET")
    NUM_SAMPLES = env.int("NUM_SAMPLES")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT)

    functions_df = pd.read_parquet(MAIN_DATASET, engine='fastparquet').sample(frac=1, random_state=42)

    # Sum of shapes[0] fits function_df.shape[0]
    multiline_functions_df, abstract_functions_df, tests_df, one_liners_df = split_into_types(functions_df)

    evaluate_and_save_results(multiline_functions_df, model, tokenizer, 'mean_values_multiline', 'result_multiline',
                              num_samples=NUM_SAMPLES)
    evaluate_and_save_results(abstract_functions_df, model, tokenizer, 'mean_values_abstract', 'result_abstract',
                              num_samples=NUM_SAMPLES)
    evaluate_and_save_results(tests_df, model, tokenizer, 'mean_values_tests', 'result_tests',
                              num_samples=NUM_SAMPLES)
    evaluate_and_save_results(one_liners_df, model, tokenizer, 'mean_values_oneline', 'result_oneline',
                              num_samples=NUM_SAMPLES)
