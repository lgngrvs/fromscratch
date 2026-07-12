# TESTING HARNESS
import tools as tls
import torch

"""

Problem: how can you verify that you've implemented things correctly?
Idea: test components against canonical pytorch implementations.
Problem: must get initialization, etc. all exactly right
TestingHarness: takes in a pytorch object and takes in a custom object and

Easy thing would be just to detect things with the same keys and set them 
to one another.

"""


