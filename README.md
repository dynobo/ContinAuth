# Continuous Authentication using Inertial-Sensors of Smartphones and Deep Learning

***Master Thesis,<br>
June 28th, 2019<br>
University of Media, Stuttgart (DE)<br>
Data Science & Business Analytics<br>***

## Errata
> If you identify any issues or irregularities in my project, [please let me know!](https://github.com/dynobo/ContinAuth/issues) I'm very eager to learn from my mistakes and will share them right here for everyone to benefit.

## Project Description
**Description**<br>
Implementation of a "Continuous Authentication" approach based on inertial sensor data of smartphones. The ensemble model, proposed by Centeno et al. (2018), consists of a Siamese CNN for deep feature learning and an OCSVM for classification. A standard OCSVM with raw data input is used as baseline.

**Goals**<br>
Attempt to reproduce results reported in the original study.<br>
Evaluate approach in scenario closer to real-world setting.<br>
Propose alternative variant of the original approach.<br>

**Timeline**<br>
Dec. 2018 - Jun. 2019

**Cite**<br>
- H. Buech (2019). "Continuous Authentication using Inertial-Sensors of Smartphones and Deep Learning". Master thesis. Hochschule der Medien, Stuttgart. URN: [ToBeDone](ToBeDone) ([BibTex](https://raw.githubusercontent.com/dynobo/ContinAuth/master/CITATION_THESIS.bib))
- H. Buech (2019). ContinAuth, GitHub repository. URL: https://github.com/dynobo/ContinAuth ([BibTex](https://raw.githubusercontent.com/dynobo/ContinAuth/master/CITATION_REPO.bib))

## Background

The legitimacy of users is of great importance for the security of information systems. The authentication process is a trade-off between system security and user experience. E.g., forced password complexity or multi-factor authentication can increase protection, but the application becomes more cumbersome for the users. Therefore, it makes sense to investigate whether the identity of a user can be verified reliably enough, without his active participation, to replace or supplement existing login processes.

This master thesis examines if the inertial sensors of a smartphone can be leveraged to continuously determine whether the device is currently in possession of its legitimate owner or by another person. To this end, an approach proposed in related studies will be implemented and examined in detail. This approach is based on the use of a so-called Siamese artificial neural network to transform the measured values of the sensors into another vector that can be classified more reliably.

## Project's directory structure

This project is structured after a setup proposed in a [tutorial on kdnuggets.com](https://www.kdnuggets.com/2018/07/cookiecutter-data-science-organize-data-project.html):

```
├── README.md          <- The top-level README for developers using this project.
├── LICENSE            <- MIT License.
├── data
│   ├── external       <- Data from third party sources. (Extracted H-MOG CSV files)
│   └── processed      <- The final, canonical data sets for modeling. (Transformed in HDF)
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks. Named after the chapter of the thesis,
│                         where the results are discussed.
│
├── reports            <- Generated exports/reports as HTML
│   ├── optimization   <- HTML exports of Notebooks during parameter tuning
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── environment.yaml   <- The environment file for reproducing the analysis
|                         environment in theAnaconda distribution by executing
|                         `conda continauth create -f environment.yaml`
|
├── pkgs.txt           <- While environment.yaml includes only explicit installed
|                         packages and should be OS agnostic, this file includes
|                         all exact dependencies, but is specific to the OS used on the
|                         developer machine. (Use this, if the one above doesn't work)
|
└── src                <- Source code for use in this project.
    │
    ├── data           <- Scripts to download or generate data
    │
    └── utility        <- Helper scripts

```

## Get it up and running
**Install environment:**
```bash
conda env create -f environment.yml
```
**Enter environment:**
```bash
conda activate continauth
```

> **NOTE:**
All following commands are expected to be executed inside this virtual environment `continauth` and with repository root (`./`) as current working directory!

**Install Jupyterlab Extensions (optional):**
```b
jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.38 @jupyterlab/toc @krassowski/jupyterlab_go_to_definition @ryantam626/jupyterlab_code_formatter jupyter-matplotlib

jupyter serverextension enable --py jupyterlab_code_formatter
```

**Download Dataset:**
- Download [H-MOG Dataset](http://www.cs.wm.edu/~qyang/hmog.html) and place the zip file as it is into the repository root.

**Run preprocessing steps:**
```bash
python -m src.data.make_dataset
```

This will run the following steps, which also can be executed manually:
- `python -m src.data.unzip_hmog_dataset` - Unzip into '/data/external' and optionally remove zip file.
- `python -m src.data.transform_to_hdf` - Reads CSVs, joins sensor data by time index and store it in HDF format in '/data/processed/' with table key `sensors_100hz`
- `python -m src.data.resample_dataset` - Reads data from HDF, resamples it to 25Hz and stores it as separate table `sensors_25hz` in the same HDF.

**Start Jupyter Lab:**
```bash
jupyter lab
```
