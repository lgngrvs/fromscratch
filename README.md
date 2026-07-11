# From Scratch

> You don't know something until you've built it from scratch

One day I would like to implement the whole language model stack from the bottom up: algorithms, architecture, optimizers, training phases, all the way down through the compiler, to the operating system. Today is not that day.

This repo contains, at the moment, bare-pytorch implementations of as many things as I can make, in a modular format that allows composability. Basically Andrej Karpathy's Micrograd/NanoGPT projects but I did them so they're different.

This repository contains *atelic* software: software built where the purpose lies in the process of building, not the end result. All code will be written by hand. 

## Rules

### Imports
The goal is to understand the components themselves; we can go to a lower level later if we want.

Allowed imports:
- `torch` (no `nn` etc.)
- `torch.einsum`: einsum makes manipulations clear

### LLM Help
All code is written by hand. Language models may be consulted as tutors but may not edit any files directly. Acceptable uses:

- They may be consulted for conceptual help, and may supply equations. Humans must turn those equations into code.
- They may be asked for debugging help, in which case they can give a line number and *if absolutely necessary* a hint at the fix, if something is broken.
- They may be asked for code feedback, **once implementation of a component is finished**. 

### Imports

All external imports route through a single main list in `tools/__init__.py`, exact package names are required so we can see every single import we use.

Most packages stay named with `tls` e.g. `tls.einsum`; some things e.g. `Module`, `Tensor` I just really don't want to have to write `tls` in front of so I'm being lazy. Maybe I will regret this at some point. Who knows!

Everything that I have built myself will have a specific directory.

### Package management 

Everything is always run through `uv`.

## Structure

I don't really know *a priori* how this is going to be structured so this will change over time, but here is my guess for now:

`/architecture`: whole architectures
`/architecture/layers.py`: layers that go into architectures (will probably get moved at some point but for now is all in one file)
`/training/scripts`: scripts for actually running 
`/training/optimizers.py`: optimizers
`/training/losses.py`: losses
`/inference`: for running inference

In pytorch, I'd like to implement architectural components, optimizers, losses, training scripts, parallelization, et cetera, and then train my own model from scratch using it all.

At some point I would like to use JAX for a mini implementation of e.g. an MLP and train it on CIFAR so that I can understand everything 'as it truly is' (in the mathematical sense) but in order to get started I am going to do it in torch.

I want everything to feel nice when importing. e.g. `from training.optimizers import cross_entropy` or something like that.


