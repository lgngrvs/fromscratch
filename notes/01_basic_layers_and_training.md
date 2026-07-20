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

## Training

Ok i implemented the training stuff. All that's left is optimization --- I think we're quite close.
Was laughing at the message that pytorch gives you when your backward() fails:

"""
112 LMFAO:
113 RuntimeError: one of the variables needed for gradient computation has been modified by an inplace         operatio
114 n: [torch.FloatTensor [1, 40, 8]], which is output 0 of AsStrided, is at version 1; expected version 0     inst
115 ead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The        variabl
116 e in question was changed in there or anywhere later. Good luck!
117
118 I think this is one very good thing for claude to help with.
119 Recognized issues:
120 - +=
121 - setting variables x[mask]=0 instead of x = x * mask
122 - masked_fill_ instead of masked_fill (inplace)
123 """

This is the kind of thing that I am really really glad I have claude for! I asked claude to find the inplace ops and claude found 3 mentioned above. Updated them and immediately it was all solved --- saved me hours probably.

Claude also helped me understand some advanced slicing stuff (but I still don't really understand it --- will require some additional thought that I don't think is worthwhile at the moment). glorious!



## OK! We have naive training running! I have added plotting, validation set testing during training, and can now train arbitrary modules using forward_with_logits and pre-one-hotting the dataset.

This is freaking awesome!! I'm having so much fun. My guys are learning!! I can't wait to do more stuff. Next thing to do is add positional encoding (lol forgot) and a learning rate scheduler most likely.

Ooooh i can creeate special tokens.... and create a chat template.... do you need RL to teach the model to use a chat template? presumably yes. that's a great toy task

so far I haven't run into any real challenges, only some bugs that I've fixed. The tokenization and embedding stuff has been moderately annoying. It's fun that such a naive setup is working (though i guess it's not really naive, the transformer is so complex... though I'm also training an MLP to do it)

I think it would be fun to create all kinds of toy tasks and see which ones the different networks can learn

i'm so damn excited this is such a good time. need to figure out what's next on the agenda


## Testing positional encoding

Ok I'm doing stupid things with the masking that are going to come back to bite me --- trying to use a new dataset but the infra does not necessarily work out of the box because i did not build the sequence stuff right

The mask stuff should probably be generated the annoying way: move the mask indices to create the labels tensor.

Hmmm ok i created a shifting helper function but I'm realizing that there are two different things that are going on: you shift and replace pad tokens for real labels, and you shift and replace mask tokens with the mask. I could give an optional argument or i could just put in a standard workflow (mask tensor is generated which pads correctly, you can influence the mask tensor. ugh this is complictaed.

what i want is:
- i have a bunch of strings; i tokenize and pad them
- during post-training i am adding a system prompt that I'm not training on. so i take my strings and i append the chat template, then i create ones_like the data, zeros_like the template on the front, zeros_like the template on the back, and then simply stick the tensors together?
- i have a template (e.g. a standardized chat template) that gets applied to text after the text is tokenized I think. might be easier to actually do the chat template in string form but I'm worried about tokenizing getting messed up or de-standardized.
- after applied, that template will give me the new batch back along with a mask tensor which has 0s in the places that the template was added


struggling to think about the masking, it's not very fun

ok i got masking working (sorta, will probably need to be rewritten)
but right now... the task that I have is tooo easy; the no-pos-encode model learns it too. even with just 1 layer!? 


2-layer transformer is needed for this it seems!

## Fixed

Ok, great. We fixed the positional dataset (finally). Now the models both fail at the task (50% accuracy) even though they're both 2-layers. As hypothesized, the model just guesses A every single time --- even though in theory it *has* positional information. 

Thus I'm going to try expanding the number of classes to prevent the degerate strategy. Looks like 2 was not enough. I'll try the full set of indices again.

OK, it turns out it actually was not fixed. I felt like I'd reach the point where the debug was not worth it, so I asked claude to go in and find it. Claude found:
1. An einsum bug (seq1 and seq2 need to be distinguished; two dimensions with the same name will use the diagnoal apparently?)
2. I was multiplying by sqrt(dk) instead of dividing by it
3. I had a bug in my attention forward that was not applying the positional encoding (LOL)
All of them were quite subtle --- probably would have each taken ~hours of debug? unsure. The einsum bug I would have eventually tested with I guess. The normalization bug would have been easiest; the positional encode one I would have found (it was a case of a misnamed variable) but it would have taken a while. Einsum one was critical though--- attention was totally broken. I don't feel too bad that I missed them all.

But even after fixing all of these, the model still couldn't get above 33%! It turns out --- it was a model capacity problem! Claude debugged further, couldn't figure it out, so Claude then just ran training runs with a larger residual stream --- fascinating!! 

Anyway--- it works!! it all seems to work!! It's time to start learning interesting things about training, gradient stability, etc. because I've been using SGD.

I think we're at the point where we can start writing notes for the blog post. This is getting fun!!


