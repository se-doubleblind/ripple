import git
import os
from pathlib import Path

import pandas as pd

from tree_sitter import Language, Parser

from git import Git, Repo


def reset_graph():
    global method_dict
    method_dict = {
        'nodes': [], 'name': [], 'path': []
    }


def add_methods_and_imports(dir_path):
    tree = lang.PARSER.parse(bytes(lang.src_code, "utf8"))
    query = lang.method_import_q
    captures = query.captures(tree.root_node)
    # adds all the method nodes to a list and all the method definition to a dictionary
    cur_method_nodes = [node[0] for node in captures if node[1] == 'method']

    method_dict['path'].extend([lang.filepath for node in cur_method_nodes])
    method_dict['nodes'].extend(cur_method_nodes)
    method_dict['name'].extend([lang.get_method_name(node) for node in cur_method_nodes])


def parse_directory(dir_path):
    reset_graph()
    if not os.path.isdir(dir_path):
        exit_with_message(f'Could not find directory: {dir_path}')

    dir_path = Path(dir_path)
    paths = dir_path.rglob(f'*{lang.extension}')
    for path in paths:
        if '/test/' in str(path) + '/':
            continue
        lang.set_current_file(str(path))
        add_methods_and_imports(str(dir_path))

    method_start_lines = [int(node.start_point[0]) + 1 for node in method_dict['nodes']]
    method_end_lines = [int(node.end_point[0]) + 1 for node in method_dict['nodes']]

    return pd.DataFrame({
        'name': method_dict['name'],
        'path': method_dict['path'],
        'start_line': method_start_lines,
        'end_line': method_end_lines,
    })


class GitHubRepository:
    def __init__(self, repo_dir, commit):
        self.repo_dir = repo_dir
        self.commit = commit
        global lang
        lang = JavaParser()

        # Checkout the commit for the repo
        g = Git(str(self.repo_dir))
        g.checkout(self.commit)
        g.clean(force=True, d=True)

        self.method_df = parse_directory(str(self.repo_dir))


class JavaParser:
    src_code = ''   #A string containing all the source code of the filepath
    lines = []      #All the lines in the current file
    filepath = ''  #the path to the current file
    extension = '.java'
    filename = 'dataset/call_graph/java-languages.so'
    language_library = Language(filename, 'java')
    PARSER = Parser()
    PARSER.set_language(language_library)
    method_import_q = language_library.query("""
            (method_declaration) @method
            (constructor_declaration) @method
            (import_declaration
                (identifier) @import)
            (import_declaration
                (scoped_identifier) @import)
            """)

    def set_current_file(self, path):
        """Sets the current file and updates the src_code and lines"""
        try:
            with open(path, 'r', encoding='utf-8', errors = 'ignore') as file:
                self.src_code = file.read()
                self.lines = self.src_code.split('\n')
                self.filepath = path
        except FileNotFoundError as err:
            print(err)

    def node_to_string(self, node) -> str:
        """Takes in a tree-sitter node object and returns the code that it refers to"""
        start_point = node.start_point
        end_point = node.end_point
        if start_point[0] == end_point[0]:
            return self.lines[start_point[0]][start_point[1]:end_point[1]]
        ret = self.lines[start_point[0]][start_point[1]:] + "\n"
        ret += "\n".join([line for line in self.lines[start_point[0] + 1:end_point[0]]])
        ret += "\n" + self.lines[end_point[0]][:end_point[1]]
        return ret

    def get_method_name(self, method):
        name = self.node_to_string(method.child_by_field_name('name'))
        return name

    def get_import_file(self, imp):
        file_to_search = self.node_to_string(imp)
        return file_to_search.replace(".", os.sep) + self.extension
