# ChoiceGP

Ranking available objects  by means of preference relations yields the most common description of individual choices. However, preference-based models assume that individuals: (1) give their preferences only between pairs of objects; (2) are always able to pick  the best preferred object.

In many situations, they may be instead choosing out of a set with more than two elements and, because of lack of information and/or incomparability (objects with  contradictory characteristics), they may not able to select a single most preferred object.  

To address these situations, we need a choice-model which allows an individual to express a set-valued choice. Choice functions provide such a mathematical framework. This is a Python implementation of a Gaussian Process model to learn choice functions from choice-data. The proposed model assumes a multiple utility representation of a choice function based on the concept of Pareto rationalization, and derives a strategy to learn both the number and the values of these latent multiple utilities.

See notebooks for examples about how to use this library.

Please if you use this software, cite it as

@inproceedings{maua2013a,
title = {Learning Choice Functions with Gaussian Processes},
author = {Benavoli, Alessio and Azzimonti, Dario and Piga, Dario},
url = {[http://www.idsia.ch/~alessio/maua2013a.pdf](https://arxiv.org/abs/2302.00406)},
year = {2023},
booktitle = {Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023)},
pages = {},
keywords = {},
pubstate = {accepted},
tppubtype = {inproceedings}
}



