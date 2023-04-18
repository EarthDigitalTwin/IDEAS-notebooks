# IDEAS Digital Twin Notebooks 

The `Flood_Demo.ipynb` notebook highlights some of the analytics and visualization capabilities of IDEAS with four use cases:
- Pakistan flooding occurring from June through September, 2022
- Hurricane Ian as it makes landfall over Florida in September, 2022
- France storm in January 2022
- LIS model data over the Mississippi river basin

The `AQACF_demo.ipynb` notebook highlights some of the analytics and visualization capabilities of AQACF with six use cases:
- 2021 Alisal Wildfire
- 2020 California Wildfires
- 2018 Carr Wildfire
- Los Angeles ports backlog Fall 2021
- Fireworks during 4th of July 2022 in Los Angeles county
- Air Pollution in the Yellow Sea
- Fires and Thermal Daily Difference in Southeast Asia in 2022

__Requirements__  

* conda >= 22.9.0  

* OS: Mac (more OS options to come)

__Running the notebook__  

To run the `Flood_Demo.ipynb` or `Generalized_AQACF.ipynb` notebook, run the following commands that create a conda environment called `ideas_notebook` using the `environment.yml` file to include all required dependencies, and install the environment as a kernelspec:
```
conda env create -f environment.yml
conda activate ideas_notebook
pip install notebook
pip install ipykernel
python -m ipykernel install --user --name=ideas_notebook
jupyter notebook
```
From the localhost page that opens, you can run the ideas notebook. Make sure you change the kernel by selecting the option at the top Kernel -> Change kernel -> ideas_notebook (see [here](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) for more information).