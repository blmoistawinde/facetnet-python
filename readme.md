## FacetNet-python
This is an unofficial implementation in python3.6 for the paper: Facetnet: a framework for analyzing communities and their evolutions in dynamic networks

`main.py`:  entrance of the program, containing 3 experiments on synthetic networks, can be run directly.

`util.py`: helper funtions of the program, including the **soft modularity** stated in the paper.

`synthetic.py`: codes for creating the synthetic networks, including the network mentioned in **part 4.1.2**.

Algorithm for both fixed and dynamic node numbers are implemented, but the one with varied community numbers is not, since I find it insufficient to implement with the information in the paper. Equation (9) and (10) would fail if the number of communities increases.
