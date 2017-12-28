# Determinantal Point Process

This repository includes sampling methods for *determinantal point processes* (*DPP*s). It is still under development.

## Prerequisites
* `Python 2.7`
* `PyTorch >= 0.2`
* `numpy`
* `scipy`
* `matplotlib`

## Currently Available Methods

* Exact (k)DPP sampling (with eigen-decomposition)
* Markov Chain Monte Carlo Sampling for (k)DPP (with Gaussian-Auadrature Acceleration)

---

## Toy Demo ('demo.py')

![](fig/unif-dpp.png)

![](fig/unif-kdpp.png)

## Demo on Nystrom Method and Approximate Kernel Ridge Regression 

![](fig/regression.png)
