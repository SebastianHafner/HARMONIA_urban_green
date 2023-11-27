
# About

This code was developed within the EU Horizon 2020 project HARMONIA with the aim to map urban green areas and detection urban green area changes in the project's pilot cities (Milan, Ixelles, Piraeus, and Sofia) using multi-spectra satellite imagery. The dataset used for model training features Landsat 8 and Sentinel-2 imagery for the reference years 2013 and 2018, respectively. Corresponding urban green labels were derived from the Urban Atlas suite.

Please follow the steps below to perform urban green mapping:

## 1 Dataset download

The HARMONIA urban green dataset can be downloaded from Zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10210237.svg)](https://doi.org/10.5281/zenodo.10210237)

## 2 Network training

To train the network, run the ``train_network.py`` file with the ``optical.yaml`` config:

````
python train_network.py -c optical -o 'path to output directory' -d 'path to the dataset' -p 'wandb project for logging'
````

## 3 Inference


Run the files ``inference.py`` for one of the four pilot cities.

````
python inference.py -c optical -s 'pilot city' -y 'reference year' -o 'path to output directory' -d 'path to the dataset'
````

# Funding
 
This work is funded by the  EU Horizon 2020 project HARMONIA (Grant agreement ID: 101003517). Please find more information on the [HARMONIA website](https://harmonia-project.eu/). 