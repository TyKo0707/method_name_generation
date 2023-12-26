from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from utils.process_dataframe import split_into_types
from evaluation_pretrained import evaluate_and_save_results

NUM_SAMPLES = 2000

checkpoint = "Salesforce/codet5p-220m"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
path_to_model = './model/'
model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model)

functions_df = pd.read_csv('datasets/functions_df_inputs_outputs.csv').sample(frac=1, random_state=42)

multiline_functions_df, abstract_functions_df, tests_df, one_liners_df = split_into_types(functions_df)

evaluate_and_save_results(multiline_functions_df, model, tokenizer, 'mean_values_multiline', 'result_multiline',
                          num_samples=NUM_SAMPLES)
evaluate_and_save_results(abstract_functions_df, model, tokenizer, 'mean_values_abstract', 'result_abstract',
                          num_samples=NUM_SAMPLES)
evaluate_and_save_results(tests_df, model, tokenizer, 'mean_values_tests', 'result_tests', num_samples=NUM_SAMPLES)
evaluate_and_save_results(one_liners_df, model, tokenizer, 'mean_values_oneline', 'result_oneline',
                          num_samples=NUM_SAMPLES)
