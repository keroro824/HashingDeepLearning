import random
lines = open('aloi.scale').readlines()
random.shuffle(lines)
open('aloi', 'w').writelines(lines)
