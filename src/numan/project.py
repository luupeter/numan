"""
Helps set up the project structure.
"""
import os


class Project:
    def __init__(self, project_folder):
        self.main_folder = project_folder

    def create(self, folder_in_project):
        path = os.path.join(self.main_folder, folder_in_project)
        assert not os.path.isdir(path), f'the directory {path} already exists in the project,' \
                                        ' did you mean to process another project?'
        os.makedirs(path)

    def activate(self, folder_in_project):
        path = os.path.join(self.main_folder, folder_in_project)
        assert os.path.isdir(path), f'Can not activate the folder {path}, the directory does not exists.'
        os.chdir(path)

    def check_exists(self, folder_in_project):
        path = os.path.join(self.main_folder, folder_in_project)
        assert os.path.isdir(path), f'Directory {path} does not exists. ' \
                                    f'Verify that this is the right project folder ' \
                                    f'and that you have run the previous analysis steps.'
