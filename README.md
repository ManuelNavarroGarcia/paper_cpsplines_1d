# paper_cpsplines_1d

`paper_cpsplines_1d` is a GitHub repository containing all the figures and
simulations results shown in the paper

```{bash}
@TECHREPORT{navarro2022,
  Author = {Navarro-Garc{\'ia}, M. and Guerrero, V. and Durban, M.},
  Title = {On constrained smoothing and out-of-range prediction using P-splines: a conic optimization approach},
  Institution = {Universidad Carlos III de Madrid},
  Address ={\url{researchgate.net/publication/347836694}},
  Year = {2022}
}
```

All the simulation studies carried out in this work use the routines implemented
in [cpsplines](https://github.com/ManuelNavarroGarcia/cpsplines), which requires
a [MOSEK](https://www.mosek.com) license to solve the optimization problems.

## Project structure

The current version of the project is structured as follows:

* **paper_cpsplines_1d**: the main directory of the project, which consist of:
  * **figures.ipynb**: A Jupyter notebook containing the code used to generate
    the figures and the tables of the paper.
  * **aux_func.py**: constituted by a collection of auxiliary functions to
    generate the figures and tables.
  * **multiple_curves.py**: contains the code to fit simultaneously multiple
    curves using the methodology proposed in Section 4.3 of the paper.
* **data**: a folder containing CSV files with simulated and real data sets.
* **img**: a folder containing the figures shown in the paper.

## Package dependencies

`paper_cpsplines_1d` mainly depends on the following packages:

* [cpsplines](https://pypi.org/project/cpsplines/).
* [Matplotlib](https://matplotlib.org/).
* [MOSEK](https://www.mosek.com). **License Required**
* [Numpy](https://numpy.org/).
* [Pandas](https://pandas.pydata.org/).

## Installation

1. To clone the repository on your own device, use

```{bash}
git clone https://github.com/ManuelNavarroGarcia/paper_cpsplines_1d.git
cd paper_cpsplines_1d
```

2. To install the dependencies, there are two options according to your
   installation preferences:

* Create and activate a virtual environment with `conda` (recommended)

```{bash}
conda env create -f env.yml
conda activate paper_cpsplines_1d
```

* Install the setuptools dependencies via `pip`

```{bash}
pip install -r requirements.txt
pip install -e .[dev]
```

3. If neccessary, add version requirements to existing dependencies or add new
   ones on `setup.py`. Then, update `requirements.txt` file using

```{bash}
pip-compile --extra dev > requirements.txt
```

and update the environment with `pip-sync`.

## Contact Information and Citation

If you have encountered any problem or doubt while using `paper_cpsplines_1d`,
please feel free to let me know by sending me an email:

* Name: Manuel Navarro García (he/his)
* Email: manuelnavarrogithub@gmail.com

If you find `paper_cpsplines_1d` or `cpsplines` useful, please cite it in your
publications.
