import matplotlib.pyplot as plt
import seaborn as sns

def get_visualization_setting():
    setting = {
        'style': 'white',
        'palette': 'deep',
        'font': 'IPAexGothic',
        'rc': {
            'figure.dpi': 300,
            'figure.figsize': (6, 4),
            'axes.spines.right': False,
            'axes.spines.top': False,
            'axes.linewidth': .8,
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linewidth': .5,
            'grid.linestyle': 'dotted',
            'axes.edgecolor': '.3',
            'axes.labelcolor': '.3',
            'xtick.color': '.3',
            'ytick.color': '.3',
            'text.color': '.3',
            'figure.constrained_layout.use': True}}
    
    return setting