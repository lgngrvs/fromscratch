# Project design notes

*Notes that I write as I am making decisions about how to do this project*


## Language choices

Choosing the language to use. Was going to default to bare pytorch. Then thought about using a lower level language and decided against it for this version of the project; would like to do lower-level stuff at some point though.

JAX is really really nice, I really want it, but I don't want to have to handle all the extra annoyingness that comes with it. On the other hand, it reveals mathematical objects as mathematical objects...

I'm going to skip for now. I'd like to do some stuff in JAX at some point because it is beautiful, but I don't want to deal with the complexity associated with it.

## Things I'd like to do in this project

- Train language model from scratch (whole process, own datasets, modern architecture, optimizations, parallelization for training at scale)

1. Basic components
2. Basic train loop
What's next? IDRK 


https://zserge.com/posts/tensor/
https://x.com/hijkzzz/status/2076843539426836874
https://x.com/Austen/status/2076745226387902925


- - -

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


