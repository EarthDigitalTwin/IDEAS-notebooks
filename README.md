# IDEAS Digital Twin Notebooks 

The `Flood_Demo.ipynb` notebook highlights some of the analytics and visualization capabilities of IDEAS with the following use cases:
- South Korea flooding in July, 2023
- Pakistan flooding occurring from June through September, 2022
- Hurricane Ian as it makes landfall over Florida in September, 2022
- France storm in January, 2022
- Flooding in Mississippi river basin in January, 2020
    - Includes 1x, 2x, and 3x precipitation scenarios, LIS NoahMP model data, and RAPID river discharge
- Flooding in Garonne in January, 2021
    - Includes 1x, 2x, and 3x precipitation scenarios, LIS NoahMP model data, RAPID river discharge, , and Telemac2d data that highlights the flow of LIS NoahMP model -> RAPID model -> Telemac2d model
__Requirements__  

* conda >= 22.9.0  

* OS: Mac (more OS options to come)

__Running the notebook__  

To run the `Flood_Demo.ipynb` notebook, run the following commands that create a conda environment called `ideas_notebook` using the `environment.yml` file to include all required dependencies, and install the environment as a kernelspec:
```
conda env create -f environment.yml
conda activate ideas_notebook
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=ideas_notebook
jupyter notebook
```
From the localhost page that opens, you can run the ideas notebook. Make sure you change the kernel by selecting the option at the top Kernel -> Change kernel -> ideas_notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).
