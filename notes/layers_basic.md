# Setting up training basic

Created 2026-07-10
Updated 2026-07-10

Starting with an `import torch`! :D

Ok I have auto differentiation. I want to set up an MLP first. I will need some training data from somewhere.

woo how do we set up a library that will be nice?

**imports**: everything gets imported through `tools.py` so i never have to handle any of that
**package management**: uv for everything 	> o < P 

i'm having anxiety over the name. i want to name it something fun. maybe i will rename it later. for now it is `fromscratch`. 

ok, ran for the first time with an error! tools needs to be a module.

Went in and handled all the pyproject.toml stuff --- just moved it to a directory in the `__init__.py` and added a note to pyproject.toml about it.

Now we're moving!

### Initialization

Realizing I need to handle initialization. I think I'm just going to default to Kaiming normal because I've heard of it, and look more into it later


Had to handle a lot of decisionmaking about repo structure.


Ah damn I think I want to have `layers.basic` and `layers.quadratic_attention` for example. whatever for now.
