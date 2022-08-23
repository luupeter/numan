import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from .notifications import slack_notification
import os


def run_notebook(notebook, notify=True, tag_users=None):
    try:
        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)

        # waits 3 hours for the cell to finish execution
        ep = ExecutePreprocessor(timeout=3 * 60 * 60, kernel_name="numan")
    except Exception as e:
        msg = f'Error before executing the notebook {notebook}\n'
        msg += e
        print(msg)
        if notify:
            slack_notification(msg, tag_users=tag_users)
        raise e

    success = True
    try:
        out = ep.preprocess(nb)
    except Exception as e:
        out = None
        msg = f'Error executing the notebook {notebook}\n' \
              f'{e}\n' \
              f'See notebook failed_{notebook} for the traceback.'
        print(msg)
        success = False
        if notify:
            slack_notification(msg, tag_users=tag_users)
        raise
    finally:
        if success:
            if notify:
                msg = f'Successfully executed the notebook {notebook}.\n\n'
                slack_notification(msg, tag_users=tag_users)
            with open(f"done_{notebook}", mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)
        else:
            with open(f"failed_{notebook}", mode='w', encoding='utf-8') as f:
                nbformat.write(nb, f)


def wait_for_manual_step(current_notebook, notebooks_before_manual,
                         notify=False, tag_users=None):
    try:
        do_wait = False
        if notebooks_before_manual is not None:
            if current_notebook in notebooks_before_manual:
                if notify:
                    msg = f"Stopping execution, waiting for you to complete the manual step."
                    slack_notification(msg, tag_users=tag_users)
                do_wait = True
    except Exception as e:
        msg = f'Error in numan.wait_for_manual_step executing the notebook {current_notebook}\n'
        msg += e
        print(msg)
        if notify:
            slack_notification(msg, tag_users=tag_users)
        raise e
    return do_wait


def run_notebooks(notebooks, notebooks_before_manual=None, notify=True, tag_users=None, rerun=False):
    if notify:
        msg = f"Started processing notebooks in {os.getcwd()}"
        slack_notification(msg, tag_users=tag_users)

    for nb_run in notebooks:
        if rerun:
            run_notebook(nb_run, notify=notify, tag_users=tag_users)
            if wait_for_manual_step(nb_run, notebooks_before_manual,
                                    notify=notify, tag_users=tag_users):
                break
        else:
            if os.path.exists(f"done_{nb_run}"):
                msg = f"Notebook done_{nb_run} already exists, skipping {nb_run}\n"
                print(msg)
                if notify:
                    slack_notification(msg, tag_users=tag_users)
                pass
            else:
                run_notebook(nb_run, notify=notify, tag_users=tag_users)
                if wait_for_manual_step(nb_run, notebooks_before_manual,
                                        notify=notify, tag_users=tag_users):
                    break
