.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/gratin.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/gratin
    
    .. image:: https://img.shields.io/coveralls/github/<USER>/gratin/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/gratin
    .. image:: https://img.shields.io/pypi/v/gratin.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/gratin/
    .. image:: https://img.shields.io/conda/vn/conda-forge/gratin.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/gratin
    .. image:: https://pepy.tech/badge/gratin/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/gratin
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/gratin


.. image:: https://readthedocs.org/projects/gratin/badge/?version=latest
    :alt: ReadTheDocs
    :target: https://gratin.readthedocs.io/en/latest/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===============================================
Gratin *(Graphs on Trajectories for Inference)*
===============================================

Gratin is a tool to characterize trajectories of random walks, i.e. motion driven by random fluctuations. This type of motion is observed at various scales and in a wide diversity of systems. 
While this package was developed for the purpose of analysing experimental data coming from photo-activated localization microscopy (PALM) experiments, nothing prevents it from being used on random walk recordings coming from other experimental setups and other domains !

To extract *summary statistics* describing trajectories, Gratin mixes two ingredients :

* an original neural network architecture using graph neural networks (GNN)
* a simulation-based inference framework

-------
Warning
-------

Gratin requires the ``pytorch-geometric`` package, whose installation depends on you CUDA version. 
Note however that you **do not need CUDA** to run Gratin, it works on CPU, it's only a bit slower. 
See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ to install it on your machine.

----------
References
----------

* Hippolyte Verdier, Maxime Duval, François Laurent, Alhassan Cassé,  Christian Vestergaard, et al.. 
  Learning physical properties of anomalous random walks using graph neural networks. 2021. : https://arxiv.org/abs/2103.11738

* Hippolyte Verdier, François Laurent, Alhassan Cassé, Christian L. Vestergaard, Christian G. Specht, Jean-Baptiste Masson
  A maximum mean discrepancy approach reveals subtle changes in α-synuclein dynamics. 2022 : https://doi.org/10.1101/2022.04.11.487825


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
