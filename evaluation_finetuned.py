from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from utils.process_dataframe import split_into_types
from evaluation_pretrained import evaluate_and_save_results
from environs import Env

env = Env()
env.read_env()
MODEL_CHECKPOINT = env.str("MODEL_CHECKPOINT")
MAIN_DATASET = env.str("MAIN_DATASET")
NUM_SAMPLES = env.int("NUM_SAMPLES")
MODEL_PATH = env.str("MODEL_PATH")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

functions_df = pd.read_parquet(MAIN_DATASET, engine='fastparquet').sample(frac=1, random_state=42)

multiline_functions_df, abstract_functions_df, tests_df, one_liners_df = split_into_types(functions_df)

evaluate_and_save_results(multiline_functions_df[50000:], model, tokenizer, 'mean_values_multiline_ft',
                          'result_multiline_ft', num_samples=NUM_SAMPLES)
evaluate_and_save_results(abstract_functions_df[3000:], model, tokenizer, 'mean_values_abstract_ft',
                          'result_abstract_ft', num_samples=NUM_SAMPLES)
evaluate_and_save_results(tests_df[7000:], model, tokenizer, 'mean_values_tests_ft', 'result_tests_ft',
                          num_samples=NUM_SAMPLES)
evaluate_and_save_results(one_liners_df[40000:], model, tokenizer, 'mean_values_oneline_ft', 'result_oneline_ft',
                          num_samples=NUM_SAMPLES)
