
Using Neuroevolution with gradient descent to construct and train logic circuits
=========================================

<p align="center">
  <img width="600" height="600" src="https://github.com/declanoller/hyper_NE_GD/blob/master/other/blog_NE_GD_cover.png">
</p>

Overview
-----------------------

This was an extension of a previous project I did where I used neuroevolution to build neural network (NN) agents to play OpenAI games. However, here, I made it so the weights aren't searched for randomly, they're trained via gradient descent, which speeds things up a ton.

In addition, I made it so a successful NN can be packaged into an "atom", to be inserted in other, higher level NNs. This part was less successful, but still has promise.

A long summary of the results can be found [on my blog here](https://www.declanoller.com/2019/05/24/descending-into-modular-neuroevolution-for-logic-circuits/).



Main run scripts
--------------------

These are the main `.py` scripts that I use, in the main directory for simplicity. They are:

* `hyper_evo.py` - For running scripts.


Classes
-----------------------

These are the classes that the run scripts use, which live in the classes directory. Here, they're listed in order from low to high level.

Software:

* `Atom.py` - Represents a single discrete entity, either a simple classic NN node, or a more complex, functional module atom.
* `HyperEPANN.py` - Assembles `Atom` objects, handles propagation through the NN, and other stuff.
* `EvoAgent.py` - Handles the NN, and interfaces with whatever environment you want to use it with (OpenAI gym, or your own).
* `Population.py` - For building a populatinon of `EvoAgent` objects. Also has `multi_run_hist()` for running the same evolution scenario multiple times, to generate statistics.







#
