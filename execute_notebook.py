"""Execute in ml_env, save partial progress and stop on cell errors."""
import os
import sys
import subprocess
from pathlib import Path
import nbformat
from nbclient import NotebookClient
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT=Path(__file__).resolve().parent
assert Path(sys.prefix).name.lower()=='ml_env'
os.chdir(ROOT)
path=ROOT/'osha_2023_eda_revisado.ipynb'
notebook=nbformat.read(path,as_version=4)
# The installed python3 kernel points to ml_env; do not create or modify a global kernel.
from jupyter_client.kernelspec import KernelSpecManager
spec=KernelSpecManager().get_kernel_spec('python3')
assert Path(spec.argv[0]).resolve()==Path(sys.executable).resolve()
notebook.metadata.kernelspec={'display_name':'Python (ml_env)','language':'python','name':'python3'}
results=ROOT/'osha_group_time_results'
def checkpoint(cell, cell_index, **kwargs):
    nbformat.write(notebook,path)
    print(f'Completed cell {cell_index}',flush=True)
client=NotebookClient(notebook,timeout=None,kernel_name='python3',allow_errors=False,
                      resources={'metadata':{'path':str(ROOT)}},on_cell_executed=checkpoint)
try:
    client.execute()
finally:
    nbformat.write(notebook,path)
print('Notebook execution completed.',flush=True)
subprocess.run([sys.executable,str(ROOT/'write_results_summary.py')],check=True)
with (results/'book_build.log').open('w',encoding='utf-8') as log:
    subprocess.run([str(Path(sys.executable).parent/'Scripts/jupyter-book.exe'),'build',str(ROOT)],
                   stdout=log,stderr=subprocess.STDOUT,check=True)
assert (ROOT/'_build/html/osha_2023_eda_revisado.html').exists()
print('Jupyter Book build completed.',flush=True)
subprocess.run([sys.executable,str(ROOT/'verify_delivery.py')],check=True)
