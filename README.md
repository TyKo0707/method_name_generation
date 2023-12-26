# Method name generation

This assignment focuses on predicting method names using a code from [IntelliJ Community project](https://github.com/JetBrains/intellij-community). 

The process involves extracting methods, adapting a pre-trained Transformer model ([CodeT5+ 220m](https://huggingface.co/Salesforce/codet5p-220m) - a state-of-the-art seq2seq pre-trained language model for code), 
evaluating predictions on collected and processed data, fine-tuning the model on a given dataset, 
and reporting changes in prediction quality.

## Steps of solution
1. **Extracting data from IntelliJ Community project**:
* Getting all Java files from the project.
* Extracting methods from obtained Java files using [tree-sitter](https://github.com/serenadeai/java-tree-sitter) library for Java language.
* Saving extracted methods to a DataFrame with following features: `function_id, identifier, formal_parameters, type_identifier, access_modifiers, block, access_modifiers_annotation, access_modifiers_test, is_abstract, is_recursive, is_test, is_multiline`.


2. **Preprocessing data for evaluation and fine-tuning**:
* Creating new DataFrame from existing one by combining some features into the whole method body but maintaining other features for further division functions into types (e.g. test, abstract, multiline, one-line).
* Version of DataFrame after previous operation has the following features: `function_id, input, label, is_multiline, is_test, is_abstract`. And it looks like this:

| function_id | input                                                                            | label                 | is_multiline   | is_test   | is_abstract   |
|------------:|:---------------------------------------------------------------------------------|:----------------------|:---------------|:----------|:--------------|
|       26835 | @TestMetadata("For2.kt") public void <extra_id_0>(  ){...Some multiline code...} | testFor2              | True           | True      | False         |

3. **Evaluating predictions using pre-trained model**:
* Using pre-trained model to predict method names for each method body with its identifier (method name) substituted by `<extra_id_0>` token.
* Calculating custom-created F1 score, accuracy and similarity metrics for each type of method (e.g. test, abstract, multiline, one-line). The obtained result is here:

| type       |   precision |   recall |   f1score |   accuracy |   similarity |
|:-----------|------------:|---------:|----------:|-----------:|-------------:|
| multiline  |    0.494142 | 0.401843 |  0.427776 |   0.401843 |     0.635965 |
| abstract   |    0.2229   | 0.137603 |  0.16287  |   0.137603 |     0.396059 |
| one-liners |    0.573125 | 0.452067 |  0.485577 |   0.452067 |     0.660239 |
| test       |    0.800313 | 0.712085 |  0.746161 |   0.712085 |     0.847186 |

4. **Fine-tuning model on a given dataset**:
* After analyzing the performance of pre-trained model and finding all its flaws of generating method names, I decided to fine-tune the model on 100k of methods from the dataset containing 50000 samples of type multiline, 3000 of abstract, 7000 of test and 40000 of one-liners.
* Creating a Dataset for fine-tuning using AutoTokenizer for Salesforce/codet5p-220m model with parameters `truncation=True, max_length=max_input_length, padding="max_length"` and `max_input_length = 512, max_target_length = 32`.
* Splitting the dataset into train and validation sets with 80/20 ratio.
* Fine-tuning the model on the train set with the following (main) parameters: `batch_size = 8, learning_rate = 4e-5, num_train_epochs = 3, eval_steps = 10000, etric_for_best_model="rouge1"`.


5. **Evaluating predictions using fine-tuned model**: 
* Same preparation of data as in step 3 (using the same data for evaluation).
* And the obtained result is:

| type       |   precision |   recall |   f1score |   accuracy |   similarity |
|:-----------|------------:|---------:|----------:|-----------:|-------------:|
| multiline  |    0.75683  | 0.691369 |  0.713991 |   0.691369 |     0.838892 |
| abstract   |    0.581858 | 0.532607 |  0.550536 |   0.532607 |     0.741023 |
| one-liners |    0.765449 | 0.722451 |  0.737629 |   0.721966 |     0.848729 |
| test       |    0.889305 | 0.838206 |  0.858557 |   0.838206 |     0.9156   |

### Executing program
I didn't create a main-file for executing the program because this doesn't really make sense but you can run all the steps using following instructions:
0. **Installing requirements**: Install all the requirements from `requirements.txt` file.
1. **Extracting data from IntelliJ Community project**: run `text_extraction.py` file with putting the path to the directory with all the Java files from IntelliJ Community project as an argument `directory`. Output will be saved to `data/functions_df.csv` file.
2. **Preprocessing data for evaluation and fine-tuning**: run `process_data_io.py` which will save the processed data to `datasets/functions_df_inputs_outputs.parquet.gz` (compress it to load it faster to google colab) file.
3. **Evaluating predictions using pre-trained model**: run `evaluation_pretrained.py`.
4. **Fine-tuning model on a given dataset**: open and run `fine_tune_notebook.ipynb` file in google colab. 
5. **Evaluating predictions using fine-tuned model**: run `evaluation_finetuned.py`.
