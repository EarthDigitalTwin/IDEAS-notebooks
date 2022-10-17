# IDEAS Digital Twin Notebooks

The `IDEAS.ipynb` notebook highlights some of the analytics and visualization capabilities of IDEAS with three usecases:
- Pakistan flooding occuring from June through September, 2022
- Hurrican Ian as it makes landfall over Florida in September, 2022
- LIS model data over the Mississippi river basin

__Requirements__  

* conda >= 22.9.0  

* OS: Mac (more OS options to come)

__Running the notebook__  

To run the `IDEAS.ipynb` notebook, run the following commands that create a conda environment called `ideas_notebook` using the `requirements.yml` file to include all required dependencies, and install the environment as a kernelspec:
```
conda env create -f requirements.yml
conda activate ideas_notebook
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=ideas_notebook
jupyter notebook
```
From the localhost page that opens, you can run the ideas notebook. Make sure you change the kernel by selecting the option at the top Kernel -> Change kernel -> ideas_notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).