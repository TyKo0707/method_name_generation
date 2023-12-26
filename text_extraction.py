import os
import pandas as pd
from tqdm import tqdm
import tree_sitter
from utils.process_dataframe import format_tuple, format_cell
from environs import Env

env = Env()
env.read_env()
LANGUAGE_BUILDER_PATH = env.str("LANGUAGE_BUILDER_PATH")
JAVA_FILES_DIRECTORY = env.str("JAVA_FILES_DIRECTORY")
UNPROCESSED_FUNCTIONS_DATASET = env.str("UNPROCESSED_FUNCTIONS_DATASET")

functions = {'identifier': [], 'formal_parameters': [],
             'type_identifier': [], 'access_modifiers': [], 'block': []}

primitive_types = ['integral', 'floating', 'boolean', 'void']


def extract_methods(node: tree_sitter.Node):
    """
    Recursively traverses a syntax tree to extract information about method declarations.

    Parameters:
    - node (tree_sitter.Node): The root node of the syntax tree.

    The function recursively traverses the syntax tree and extracts information about method declarations,
    including identifiers, formal parameters, type identifiers, access modifiers, and code blocks.

    Does not return anything, but updates the global variable 'functions'.
    """
    if node.type == 'method_declaration':
        curr_function = {'identifier': [], 'formal_parameters': [], 'type_identifier': [], 'access_modifiers': [],
                         'block': [], }
        for child in node.children:
            if child.type == 'modifiers' or child.type == 'modifier':
                curr_function['access_modifiers'] = [mod.text.decode('utf8') for mod in child.children]
            elif child.type in ['block', 'formal_parameters', 'type_identifier', 'identifier']:
                curr_function[child.type].append(child.text.decode('utf8'))
                continue
            elif 'type' in child.type and (child.type.split('_')[0] in primitive_types or child.type == 'generic_type'):
                curr_function['type_identifier'].append(child.text.decode('utf8'))

        if not curr_function['type_identifier']:
            curr_function['type_identifier'].append('void')

        for key in curr_function.keys():
            functions[key].append(curr_function[key])

    for child in node.children:
        extract_methods(child)


def read_file(file_path: str) -> str:
    """
    Reads a file and returns its contents as a string.

    Parameters:
    - file_path (str): The path to the file to be read.

    Returns:
    - file_body (str): The contents of the file as a string.
    """
    encodings_to_try = ['utf-8', 'ISO-8859-1', 'windows-1252', 'utf-16']

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                file_body = file.read()
            break
        except UnicodeDecodeError:
            continue
    return file_body


def extract_functions_from_file(file_path: str):
    """
    Extracts functions from a file and updates the global variable 'functions'.

    Parameters:
    - file_path (str): The path to the file to be read.
    """
    code = read_file(file_path)
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root_node = tree.root_node
        extract_methods(root_node)
    except:
        pass


def find_java_files(global_repository_directory: str) -> list[str]:
    """
    Finds all Java files in a directory and its subdirectories.

    Parameters:
    - global_repository_directory (str): The path to the directory to be searched.

    Returns:
    - res (list[str]): A list of paths to Java files.
    """
    res = []
    for (dir_path, dir_names, file_names) in os.walk(global_repository_directory):
        res.extend([os.path.join(dir_path, file_name) for file_name in file_names if file_name.endswith(".java")])
    return res


if __name__ == "__main__":
    # To use a tree-sitter library, you need to copy it
    # from github (https://github.com/tree-sitter/tree-sitter-java) to your working directory
    tree_sitter.Language.build_library(LANGUAGE_BUILDER_PATH, ['tree-sitter-java'])
    JAVA_LANGUAGE = tree_sitter.Language(LANGUAGE_BUILDER_PATH, 'java')
    parser = tree_sitter.Parser()
    parser.set_language(JAVA_LANGUAGE)

    java_files = find_java_files(JAVA_FILES_DIRECTORY)

    for i in tqdm(range(len(java_files))):
        extract_functions_from_file(java_files[i])

    df = pd.DataFrame(functions)

    # Create strings from lists
    df = df.map(format_cell)

    # Format formal parameters column
    df['formal_parameters'] = df['formal_parameters'].apply(lambda x: format_tuple(x))

    # Remove rows with empty access modifiers
    df = df[df.access_modifiers != '']

    # Extract annotations from access modifiers
    df['access_modifiers_annotation'] = df.access_modifiers \
        .apply(lambda x: [modifier.strip() for modifier in x.split(', ') if not modifier.startswith('@Test') and
                          modifier.startswith('@')])

    # Extract tests from access modifiers
    df['access_modifiers_test'] = df.access_modifiers \
        .apply(lambda x: [modifier.strip() for modifier in x.split(', ') if modifier.startswith('@Test')])

    df.access_modifiers = df.access_modifiers \
        .apply(lambda x: ', '.join([modifier.strip() for modifier in x.split(', ') if not modifier.startswith('@')]))

    # Add 'is_abstract', 'is_recursive', 'is_test' and 'is_multiline' binary columns
    df['is_abstract'] = df.access_modifiers.apply(lambda x: 'abstract' in x.split(', '))
    df['is_recursive'] = df.apply(lambda row: f" {row.identifier}(" in str(row.block), axis=1)
    df['is_test'] = df.access_modifiers_test.apply(lambda x: len(x) > 0)
    mask = df['block'].str.contains('return') == False
    df['is_multiline'] = (df['block'].str.count('\n') > 2) | ((df['block'].str.count('\n') == 2) & mask)

    # Remove rows with empty access modifiers after creating 'access_modifiers_annotation'
    df = df[df.access_modifiers != '']

    # Create strings from lists (for access_modifiers_annotation)
    df = df.map(format_cell)

    df = df.reset_index(drop=False)
    df = df.rename(columns={'index': 'function_id'})
    df.to_csv(UNPROCESSED_FUNCTIONS_DATASET, index=True)
