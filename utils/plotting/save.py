import os


def save_plot(figure, path):
    split = os.path.split(path)
    if split[0] != '':
        os.makedirs(split[0], exist_ok=True)
    figure.savefig(path, format='pdf')
    pass
