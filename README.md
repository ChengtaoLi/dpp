# Determinantal Point Process

This repository includes sampling methods for *determinantal point processes* (*DPP*s). It is still under development.

## Prerequisites
* `Python 2.7`
* `numpy`
* `scipy`
* `matplotlib`
* `Matlab`

## Currently Available Methods

* (Python) Exact (k)DPP sampling (with eigen-decomposition)
* (Python) Markov Chain Monte Carlo Sampling for (k)DPP (with Gaussian-Auadrature Acceleration)
* (Matlab) Markov Chain Monte Carlo Sampling for (k)DPP (with Gaussian-Auadrature Acceleration)

---

## Toy Demo ('demo.py')

![](fig/unif-dpp-mcdpp.png)

![](fig/unif-kdpp-mckdpp.png)

## Demo on Nystrom Method

![](fig/nystrom.png)

## Demo on Approximate Kernel Ridge Regression 

![](fig/regression.png)

## Demo on Approximate Kernel Logistic Regression

![](fig/classification.png)
