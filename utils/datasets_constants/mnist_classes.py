"""MNIST class names for CLIP zero-shot.

The digit glyph alone ("0") is a weak text prompt, so the spelled-out word is used, matching the
convention CLIP's own MNIST evaluation follows.
"""

mnist_classes = ["zero", "one", "two", "three", "four",
                 "five", "six", "seven", "eight", "nine"]

nr_of_classes = len(mnist_classes)
mnist_elements_per_class = 1000
