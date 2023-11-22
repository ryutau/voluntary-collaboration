# Voluntary collaboration

This repository contains code and data for the paper "Voluntary participation and optimism: How and when outside individual options can facilitate group collaboration"
by Ryutaro Mori, Nobuyuki Hanaki, and Tatsuya Kameda

## Overview
This repository is organized into the following subdirectories:

- `data` contains original data from the experiment.
- `src` contains Python scripts for analyses and visualizations in the paper, with the help of several functions in `tools`.
- `documents` includes the English translation of experimental materials.

## Set up
The project uses Python 3. One way to set it up is by using pyenv (you may otherwise use virtualenv):
```
$ brew install pyenv # You may need to install brew beforehand
$ pyenv install 3.10.11
$ pyenv global 3.10.11
```
Please also clone this repository to your local environment:
```
$ git clone git@github.com:ryutau/voluntary-collaboration.git
```

Then, go to the directory and pip install some libraries using `requirements.txt`:
```
$ cd voluntary-collaboration
$ pip install -r requirements.txt
```

## Running analyses
After setting up the environment, navigate to the `src` directory to run various analysis scripts and reproduce the paper's results:
```
$ cd src
```

First, please run `src/main_analysis.py` to perform all the analysis. This may take few hours depending on your environment.
```
$ python main_analysis.py
```
Output files are saved in the `src/output` folder. To review regression analysis, directly inspect `bootstrap_reg_stats.csv`.


Then, you can run `src/fig~~.py` to reproduce figures in the paper. Each script will not take more than a few minutes. For example, to create figure 1:
```
$ python fig_1.py
```
Output figures are stored under the `src/output` folder.

## Further reference
For further reference, please contact to [Ryutaro Mori](https://ryutau.github.io/) (ryutau.mori[at]gmail.com).

This project is also linked to a [preprint](https://www.researchsquare.com/article/rs-3300738/v3) and an [OSF repository](https://osf.io/2cn56/), which includes a preregistlation file.
