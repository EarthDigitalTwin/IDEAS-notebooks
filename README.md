# IDEAS Digital Twin Notebooks

The `IDEAS.ipynb` notebook highlights some of the analytics and visualization capabilities of IDEAS with three usecases:
- Pakistan flooding occuring from June through September, 2022
- Hurrican Ian as it makes landfall over Florida in September, 2022
- LIS model data over the Mississippi river basin

## Running the notebook

To run the `IDEAS.ipynb` notebook, create a conda environment using the `requirements.yml` file:
```
conda env create -f requirements.yml
```
This will create a conda environment named `ideas_notebook` which will include all required dependencies. You can use this environment as the Jupyter kernel to run the notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).


## Troubleshooting
If you are having trouble launching the notebook and get the error message "Jupyter command `jupyter-notebook` not found.", try running `pip install notebook` and then `jupyter notebook`