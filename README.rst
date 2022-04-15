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
    :target: https://gratin.readthedocs.io/en/stable/

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

===============================================
GRATIN *(Graphs on Trajectories for Inference)*
===============================================

Gratin is an analysis tool for stochastic trajectories, based on graph neural networks.

First, each trajectory is turned into a graph, in which positions are nodes, and edges are drawn between them following a pattern based on their time difference. 

Then, features computed from normalized positions are attached to nodes : cumulated distance covered since origin, distance to origin, maximal step size since origin... 

These graphs are then passed as input to a graph convolution module (graph neural network), which outputs, for each trajectory, a latent representation in a high-dimensional space. 

This fixed-size latent vector is then passed as input to task-specific modules, which can predict the anomalous exponent or the random walk type. Several output modules can be trained at the same time, using the same graph convolution module, by summing task-specific losses. 

The model can receive trajectories of any size as inputs. The high-dimensional latent representation of trajectories can be projected down to a 2D space for visualisation and provides interesting insights regarding the information extracted by the model (see details in the paper).

-------
Warning
-------

Gratin relies on the pytorch-geometric package. 
See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html>`_ to install it on your machine.

---------
Reference
--------- 

- Hippolyte Verdier, Maxime Duval, François Laurent, Alhassan Cassé,  Christian Vestergaard, et al.. 
  Learning physical properties of anomalous random walks using graph neural networks. 2021. : https://arxiv.org/abs/2103.11738

- Hippolyte Verdier, François Laurent, Alhassan Cassé, Christian L. Vestergaard, Christian G. Specht, Jean-Baptiste Masson
  A maximum mean discrepancy approach reveals subtle changes in α-synuclein dynamics. 2022 : https://doi.org/10.1101/2022.04.11.487825


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
