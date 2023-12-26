import pandas as pd
from utils.code_generation import extract_tokens_from_camel
from utils.metrics import code_gen_f1_score_and_accuracy, code_gen_name_similarity
from tqdm import tqdm


def evaluate_and_save_results(df, model, tokenizer, mean_values_file_name, result_file_name, num_samples=1000):
    result = predict_and_evaluate(model, tokenizer, f'datasets/{mean_values_file_name}.csv', dataset=df[:num_samples])
    result.to_csv(f'datasets/{result_file_name}.csv', index=True)
    mean_values = result[['precision', 'recall', 'f1_score', 'accuracy', 'similarity']].mean(axis=0)
    mean_values.to_csv(f'datasets/{mean_values_file_name}.csv', index=True)


def evaluate_output(generated_result, reference, tokenizer):
    decoded_result = tokenizer.decode(generated_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    try:
        final = max(decoded_result.split(), key=len)
    except:
        final = ''

    prediction = extract_tokens_from_camel(final)
    metrics = code_gen_f1_score_and_accuracy(prediction, reference)
    metrics = metrics + (code_gen_name_similarity(prediction, reference),)
    return metrics, final


def intermediate_save(df, index, interval_value, file_name):
    if (index + 1) % interval_value == 0:
        result = df[:index][['precision', 'recall', 'f1_score', 'accuracy', 'similarity']].mean(axis=0)
        result.to_csv(file_name, index=True)


def predict_and_evaluate(model, tokenizer, mean_values_file_name, dataset=None):
    if isinstance(dataset, pd.DataFrame):
        result_df = dataset.copy()

        # Initialize new columns
        new_columns = ['prediction', 'precision', 'recall', 'f1_score', 'accuracy', 'similarity']
        result_df[new_columns] = 0.0

        columns_to_remove = dataset.columns.values.tolist()
        columns_to_remove.remove('label')
        result_df.drop(columns=columns_to_remove, inplace=True)
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
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    from utils.process_dataframe import split_into_types

    NUM_SAMPLES = 2000

    checkpoint = "Salesforce/codet5p-220m"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    functions_df = pd.read_parquet('datasets/functions_df_inputs_outputs.parquet.gz', engine='fastparquet')\
        .sample(frac=1, random_state=42)

    # Sum of shapes[0] fits function_df.shape[0]
    multiline_functions_df, abstract_functions_df, tests_df, one_liners_df = split_into_types(functions_df)

    evaluate_and_save_results(multiline_functions_df, model, tokenizer, 'mean_values_multiline', 'result_multiline',
                              num_samples=NUM_SAMPLES)
    evaluate_and_save_results(abstract_functions_df, model, tokenizer, 'mean_values_abstract', 'result_abstract',
                              num_samples=NUM_SAMPLES)
    evaluate_and_save_results(tests_df, model, tokenizer, 'mean_values_tests', 'result_tests', num_samples=NUM_SAMPLES)
    evaluate_and_save_results(one_liners_df, model, tokenizer, 'mean_values_oneline', 'result_oneline',
                              num_samples=NUM_SAMPLES)
