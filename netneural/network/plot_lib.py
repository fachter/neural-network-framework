from matplotlib import pyplot as plt


def plot_line(list_to_plot, x_label, y_label, title):
    fig, ax = plt.subplots()
    ax.plot(list_to_plot)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True);
    fig.show()


def plot_scatter(x_data, y_data, colors, x_label, y_label, title):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.scatter(x_data, y_data, c=colors)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
    fig.show()
