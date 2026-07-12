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


### Code editor

I tried switching code editors to VSCodium because I wanted things to run more smoothly --- mainly I wanted a nice run panel. But I got stressed out. I find modern code editors really irritating --- I want an extremely specific design and can't get it and then get stuck trying to handle the editor.

One day I'll get there.


Anyway, I built a bunch of stuff. I finished attention, built a transformer template, and now am building toy tokenizers and embedding layers so we can have a real end-to-end transformer!

I don't really know how to validate my implementations of some of the components...


Alright! Well, I have a transformer that runs without bugs. That does not mean a transformer that *works*, because there's a good shot I have some error sequestered in there --- most likely in the einsums, if i had to bet. There's just no way that einsums are that magical, and I feel like most likely I got something wrong somewhere.

I've been working for like 5h or something. Spent soo much time today on this and I'm really having a good time. It's incredibly satisfying.

Looking forward to cleaning up the files and running training soon :D
