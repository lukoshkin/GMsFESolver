import re
import sys
sys.path.append('..')

from gmsfem.subdiv import triplets

with open('cluster.py', 'r') as fp:
    content = fp.read()

n_el = re.search(r'n_el = ([0-9]+)', content).group(1)
n_blocks = re.search(r'n_blocks = ([0-9]+)', content).group(1)
n_el, n_blocks = map(int, [n_el, n_blocks])

triplets(n_el, n_blocks)
