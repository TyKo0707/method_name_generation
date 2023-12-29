# Method name generation

This assignment focuses on predicting method names using code from the [IntelliJ Community Project](https://github.com/JetBrains/intellij-community). 

The process involves extracting methods, adapting a pre-trained Transformer model ([CodeT5+ 220m](https://huggingface.co/Salesforce/codet5p-220m) - a state-of-the-art seq2seq pre-trained language model for code), 
evaluating predictions on collected and processed data, fine-tuning the model on a given dataset, 
and reporting changes in prediction quality.

Why [CodeT5+ 220m](https://huggingface.co/Salesforce/codet5p-220m) was chosen: it suits the best for generating the exact chosen part of a code, and its train time is exponentially smaller than other models'.

## Steps of solution
1. **Extracting data from the IntelliJ Community Project**:
* Getting all Java files from the project.
* Extracting methods from obtained Java files using the [tree-sitter](https://github.com/serenadeai/java-tree-sitter) library for Java.
* Saving extracted methods to a DataFrame with the following features: `function_id, identifier, formal_parameters, type_identifier, access_modifiers, block, access_modifiers_annotation, access_modifiers_test, is_abstract, is_recursive, is_test, is_multiline`.


2. **Preprocessing data for evaluation and fine-tuning**:
* Creating a new DataFrame from an existing one by combining some features into the whole method body but maintaining other features for further division functions into types (e.g., test, abstract, multiline, one-line).
* The version of DataFrame after previous operation has the following features: `function_id, input, label, is_multiline, is_test, is_abstract`. And it looks like this:

| function_id | input                                                                            | label                 | is_multiline   | is_test   | is_abstract   |
|------------:|:---------------------------------------------------------------------------------|:----------------------|:---------------|:----------|:--------------|
|       26835 | @TestMetadata("For2.kt") public void <extra_id_0>(  ){...Some multiline code...} | testFor2              | True           | True      | False         |

3. **Evaluating predictions using a pre-trained model**:
* Using a pre-trained model to predict method names for each method body with its identifier (method name) substituted by `<extra_id_0>` token.
* Calculating custom-created F1 score, accuracy and similarity metrics for each type of method (e.g., test, abstract, multiline, one-line). The obtained result is here:

| type       |   precision |   recall |   f1score |   accuracy |   similarity |
|:-----------|------------:|---------:|----------:|-----------:|-------------:|
| multiline  |    0.494142 | 0.401843 |  0.427776 |   0.401843 |     0.635965 |
| abstract   |    0.2229   | 0.137603 |  0.16287  |   0.137603 |     0.396059 |
| one-liners |    0.573125 | 0.452067 |  0.485577 |   0.452067 |     0.660239 |
| test       |    0.800313 | 0.712085 |  0.746161 |   0.712085 |     0.847186 |

4. **Fine-tuning model on a given dataset**:
* After analyzing the performance of the pre-trained model and finding all its flaws in generating method names, I decided to fine-tune the model on 100k methods from the dataset containing 50000 samples of multiline, 3000 of abstract, 7000 of test, and 40000 of one-liners.
* Creating a Dataset for fine-tuning using AutoTokenizer for Salesforce/codet5p-220m model with parameters `truncation=True, max_length=max_input_length, padding="max_length"` and `max_input_length = 512, max_target_length = 20`.
* Splitting the dataset into training and validation sets with an 80/20 ratio.
* Fine-tuning the model on the train set with the following (main) parameters: `batch_size = 8, learning_rate = 4e-5, num_train_epochs = 3, eval_steps = 10000, etric_for_best_model="rouge1"`.


5. **Evaluating predictions using a fine-tuned model**: 
* Same preparation of data as in step 3 (using the same data for evaluation).
* And the obtained result is:

| type       |   precision |   recall |   f1score |   accuracy |   similarity |
|:-----------|------------:|---------:|----------:|-----------:|-------------:|
| multiline  |    0.75683  | 0.691369 |  0.713991 |   0.691369 |     0.838892 |
| abstract   |    0.581858 | 0.532607 |  0.550536 |   0.532607 |     0.741023 |
| one-liners |    0.765449 | 0.722451 |  0.737629 |   0.721966 |     0.848729 |
| test       |    0.889305 | 0.838206 |  0.858557 |   0.838206 |     0.9156   |

## Executing program
I didn't create a main file for executing the program because this doesn't really make sense, but you can run all the steps using the following instructions:


0. **Installing requirements and setting env-variables**: Install all the requirements from the `requirements.txt` file. You also need to assign environment variables in the following way:
```
LANGUAGE_BUILDER_PATH='/Users/user/path_to_project/build/my-languages.so'
JAVA_FILES_DIRECTORY='/Users/user/path_to_project/intellij-java-files'
UNPROCESSED_FUNCTIONS_DATASET='/Users/user/path_to_project/datasets/functions_df.csv'
MODEL_CHECKPOINT='Salesforce/codet5p-220m'
MAIN_DATASET='/Users/user/path_to_project/datasets/functions_df_inputs_outputs.parquet.gz'
MODEL_PATH='/Users/user/path_to_project/model/'
NUM_SAMPLES=2000
```
1. **Extracting data from the IntelliJ Community Project**: run the `text_extraction.py` file with the path to the directory with all the Java files from the IntelliJ Community project as an argument `JAVA_FILES_DIRECTORY`. Output will be saved to the `datasets/functions_df.csv` file.
2. **Preprocessing data for evaluation and fine-tuning**: run `process_data_io.py` which will save the processed data to `datasets/functions_df_inputs_outputs.parquet.gz` (compress it to load it faster to google colab) file.
3. **Evaluating predictions using a pre-trained model**: run `evaluation_pretrained.py`.
4. **Fine-tuning model on a given dataset**: open and run `fine_tune_notebook.ipynb` file in google colab. You also can download it by this link to my [Google Drive](https://drive.google.com/drive/folders/1REJ0zI3oeYOZCpBWYlT2IsDO7Md-C-r-?usp=sharing). If you want to use it, put the model folder in the main directory of the project.
5. **Evaluating predictions using a fine-tuned model**: run `evaluation_finetuned.py`.


## Conclusion
**Conclusion:**

1. **Evaluating the pre-trained model:**
   - Results are not satisfactory, with poor predictions, especially in certain types of functions.
   - Types of errors in prediction have been thoroughly analyzed [here](https://github.com/TyKo0707/method_name_generation/blob/main/eda_plus_analysis.ipynb).

2. **Fine-tuning the model:**
   - CodeT5+ 220m model was selected for fine-tuning.
   - Parameters and metrics for fine-tuning were chosen based on online resources and logical reasoning.

3. **Evaluating the fine-tuned model:**
   - The fine-tuning process led to a significant improvement in prediction quality.
   - All types of functions showed increased values in metrics, including precision, recall, F1 score, accuracy, and similarity.
   - Errors in predictions were also analyzed [here](https://github.com/TyKo0707/method_name_generation/blob/main/eda_plus_analysis.ipynb) for the fine-tuned model.

In summary, the fine-tuning process successfully addressed the limitations of the pre-trained model, 
resulting in improved performance across different types of functions. 
The approach taken, involving data extraction, preprocessing, and careful evaluation, 
demonstrates a systematic effort to enhance the model's capability in predicting method names from code snippets.
A clear example of the result can be shown using these two plots:
![1](https://github.com/TyKo0707/method_name_generation/assets/65500151/dc28c316-9422-47d4-8bff-143323be3897)
![2](https://github.com/TyKo0707/method_name_generation/assets/65500151/4d06ad53-07d9-43af-821f-b0de813ffde5)

## Other ideas and further work
1. We need to consider other approaches for this task that use neural networks and attention mechanisms ([Structured neural summarization](https://arxiv.org/pdf/1811.01824.pdf)) or delve into code structure with creating some new substructures of it ([code2seq](https://arxiv.org/pdf/1808.01400.pdf)), approaches based on generated code keywords ([Keywords Guided Method Name Generation](https://arxiv.org/pdf/2103.11118.pdf)), etc.
2. Clearly, we can improve our model by obtaining more data and training the model on it.
3. We can try to combine it with XAI (make the prediction generation explainable) and try to perform knowledge distillation.
4. ...
