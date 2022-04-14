# Dragonnet reproduced

This repository contains software and data reproducing the dragonnet implementation by ["Shi et al."](https://github.com/claudiashi57/dragonnet) This is done through rewriting certain parts of their implementation with PyTorch. The paper that belongs to this repository is ["Adapting Neural Networks for the Estimation of Treatment Effects"](arxiv:https://arxiv.org/abs/1906.02120).
The paper describes the use of neural networks for the estimation of treatment effects from observational data.



# Requirements and setup
You will need to install sklearn, numpy, pytorch and, pandas

# Data

1. IHDP

* This dataset is based on a randomized experiment investigating the effect of home visits by specialists on future cognitive scores.
* It is generated via the [npci package](https://github.com/vdorie/npci).
* We also uploaded a portion of the simulated data in the dat folder.


2. ACIC

* ACIC is a collection of semi-synthetic datasets derived from the linked birth and infant death data (LBIDD)
Here is the full dataset description [data set description](https://www.researchgate.net/publication/11523952_Infant_Mortality_Statistics_from_the_1999_Period_Linked_BirthInfant_Death_Data_Set)

* Here is the GitHub repo associated with the competition [IBM benchmark](https://github.com/IBM-HRL-MLHLS/IBM-Causal-Inference-Benchmarking-Framework/blob/master/data/LBIDD/scaling_params.csv)
For access to the ACIC 2018 competition data: Please see here [ACIC data] (https://www.synapse.org/#!Synapse:syn11294478/wiki/486304)

# Reproducing the IHDP experiments
The workflow consists of two stages:

1. Fitting a predictor:

* You'll run the from `src` code as `./experiment/run_ihdp.sh`
* If you are using a cluster, there's some code that might be useful in the submission folder. 

* Before doing this, you'll need to edit `run_ihdp.sh` and change the following:
`data_base_dir= where you stored the data`
`output_base_dir=wherer you want the result to be`

* The prediction should go into your output folder

2. Estimation:
* After you get the predictions, you want to fit them into estimators. 
* Run `ihdp_ate.py` to reproduce the table. 

3. Things to note when reproducing the result: 

* The default setting would let you run Dragonnet, and TARNET under both targeted regularization and default mode. If you want to run a subset of the models, delete them at the `run_ihdp.sh`

* The default use all the data for training, prediction, and estimation. To change that, you could update the train_test_split function in the ihdp_main.py


# Reproducing the ACIC experiment
1. Fitting a predictor:
Same as above except you run the from `src` code as `./experiment/run_acic.sh`

2. Estimation:
Same as above except you run the from `src` code as `acic_ate.py`

# Contact us
This repository is created by Alec Nonnemaker, Sara Boby and Fernando Corte Vargas.
Feel free to email us if you have any questions: alec.michael@live.nl






