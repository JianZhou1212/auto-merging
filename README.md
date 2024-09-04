# Uncertainty-Aware Decision-Making and Planning for Autonomous Forced Merging
This is the Python code for the article which is published in Proceedings of 63rd IEEE Conference on Decision and Control, 2024.
```
@inproceedings{zhou2024uncertianty,
  title={Uncertainty-Aware Decision-Making and Planning for Autonomous Forced Merging},
  author={Zhou, Jian and Gao, Yulong and Olofsson, Bj\"orn and Frisk, Erik},
  booktitle={2024 63nd IEEE Conference on Decision and Control (CDC)},
  pages={ },
  year={2024},
  organization={IEEE}
}
```

Jian Zhou and Erik Frisk are with the Department of Electrical Engineering, Linköping University, Sweden. Yulong Gao is with the Department of Electrical and Electronic Engineering, Imperial College London, UK. Björn Olofsson is with both the Department of Automatic Control, Lund
University, Sweden, and the Department of Electrical Engineering, Linköping University, Sweden.

For any questions, feel free to contact me as: jian.zhou@liu.se or zjzb1212@qq.com
## Packages for running the code
To run the code you need to install the following key packages:

**Pytope**: https://pypi.org/project/pytope/

**CasADi**: https://web.casadi.org/

**HSL Solver**: https://licences.stfc.ac.uk/product/coin-hsl

Note: Installing the HSL package can be a bit comprehensive, but the solvers just speed up the solutions. You can comment out the places where the HSL solver is used (i.e., comment out the command "ipopt.linear_solver": "ma57"), and just use the default linear solver of ipopt in CasADi.

## Introduction to the files
In the folder `Case_1`, you will find the implementation for comparing the proposed method, the DMPC, and RMPC, in a single scenario. The file `main.ipynb` defines the main file for running the code; `Planner_P` is the method for the proposed approach, `Planner_D` is the method for the DMPC, and `Planner_R` is the method for the RMPC. The two mat files `xi_HD.mat` and `cdf_HD.mat` are the distribution data, which is identified offline using the highD dataset, to simulate the control input of SVs. The folders `SVModelingHighDDataDistribution` contains the method for modeling the SV0 and SV1 that are driven by the highD data.

After running the file `main.ipynb`, you should be able to generate the simulation animations. The analysis of the results will be straightforward using the output of each method.










