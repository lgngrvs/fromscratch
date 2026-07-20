# Training improvements (Note 02)


To-do list:
- [ ] Clean up train_naive.py
- [ ] Fix gradient stability issues (seems like things tend to explode!)
- [ ] Create a learning rate schedule
- [ ] Add additional optimizers
- [ ] Start writing a blog post about this basic task


This is the first set of notes that I think I want to turn into a blog post.
Ok! So here's where we are at (JUL 20): 
- We have just gotten positional encoding working correctly, and confirmed that the model can learn our toy task
- How much capacity does it need? We can run a sweep to find out
- In the process, we are learning about how to handle gradient explosion issues! This is cool --- our first real ML engineering challenge, just from this really basic setup! I'm excited!

I think the first thing to do is to figure out the gradient explosion. My plan is to add a trigger to the training code that if the gradient is nan you stop training, print out as much info as you can, and start diagnosing that way. I'll figure out a plan from there.

Here are the current loss graphs:

**Model with positional encoding**
![[images/02lossgraph1.png]]

**Model without positional encoding**
![[images/02lossgraph2.png]]

So what we're seeing is that (1) there's something wrong with the gradient — frequent explosions of the loss during training, and the non-pos encode model does actually just die. The loss becomes `nan` for some reason. Why is this?

When the `without_positional_encoding` model survives, we get this very cool graph: 
![[images/02withoutposencodesurvive.png]]
So fascinating and odd!