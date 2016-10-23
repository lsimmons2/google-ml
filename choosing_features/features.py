import numpy as np
import matplotlib.pyplot as pyplot

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

pyplot.hist([grey_height, lab_height], stacked=True, color=['r', 'b'])
pyplot.savefig('./features.png')
