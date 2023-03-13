import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import check_matplotlib_support
from itertools import product

def plot_projection(projection, labels=None, ax=None, legend_labels = ["STAR", "GAL", "QSO"], color_palette = ["k", "orange", "blue"]):
    # print(plt.rcParams['backend'])

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(w = 5, h = 5)
    else:
        fig = ax.get_figure()

    # load data
    if type(projection) == str:
        projection = np.loadtxt(projection, dtype=np.float32)

    if type(labels) == str:
        labels = np.loadtxt(labels, dtype=np.uint8)
    
    show_legend = True
    if labels is None:
        labels = np.zeros(projection.shape[0])
        show_legend = False

    # combine into pandas dataframe
    df = pd.DataFrame(data=np.column_stack((projection, labels)), columns=["y1", "y2", "Hclass"])
    del projection; del labels

    # plot data
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue="Hclass", palette=color_palette, s=2, ax=ax)
                
    # edit legend
    if show_legend:
        h, l = ax.get_legend_handles_labels()
        ax.legend(handles=h, labels=legend_labels)

    fig.tight_layout()

    return ax

def plot_projection_grid(projection, labels=None, legend_labels = ["STAR", "GAL", "QSO"], color_palette = ["k", "orange", "blue"]):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(w = 10, h = 10)

    # load data
    if type(projection) == str:
        projection = np.loadtxt(projection, dtype=np.float32)

    if type(labels) == str:
        labels = np.loadtxt(labels, dtype=np.uint8)
    
    show_legend = True
    if labels is None:
        labels = np.zeros(projection.shape[0])
        show_legend = False

    # combine into pandas dataframe
    df = pd.DataFrame(data=np.column_stack((projection, labels)), columns=["y1", "y2", "Hclass"])
    del projection; del labels

    # plot data
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], hue="Hclass", palette=color_palette, s=2, ax=axs[0,0])

    known_labels = np.unique(df["Hclass"].values)
    for i, ax in enumerate(axs.ravel()[1:]):
        data = df[df["Hclass"] == known_labels[i]]
        sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue="Hclass", palette=[color_palette[i]], s=2, ax=ax)
                
    # edit legend
    if show_legend:
        h, l = axs[0,0].get_legend_handles_labels()
        axs[0,0].legend(handles=h, labels=legend_labels)

        for i, ax in enumerate(axs.ravel()[1:]):
            h, l = ax.get_legend_handles_labels()
            ax.legend(handles=h, labels=[legend_labels[i]])


    # edit axis limits
    xlim, ylim = axs[0,0].get_xlim(), axs[0,0].get_ylim()
    for ax in axs.ravel()[1:]:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()

    return fig

def plot_shepard_diagram(SD, ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(w = 5, h = 5)
    else:
        fig = ax.get_figure()

    # ax.set_aspect('equal')
    ax.set_title("Shepard Diagram")
    ax.set_xlabel(r"$\left\| \mathbf{x}_i - \mathbf{x}_j \right\|$")
    ax.set_ylabel(r"$\left\| P\left(\mathbf{x}_i\right) - P\left(\mathbf{x}_j\right)\right\|$")

    # ax.scatter(SD[:,0], SD[:,1], c='k', s=.1)
    hbin = ax.hexbin(SD[:,0], SD[:,1], gridsize=100, bins='log', mincnt=1, cmap='Greys')
    cbar = fig.colorbar(hbin, ax=ax)
    cbar.set_label(r"$N$")

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    x = np.linspace(*ax.get_xlim(), 10)

    # compute the distance scaling
    scale = np.sum(SD[:,0] * SD[:,1]) / np.sum(SD[:,0] ** 2)

    # compute location of the annotating arrow
    if scale > ax.get_ylim()[1] / ax.get_xlim()[1]:
        xloc_a = ax.get_ylim()[1] / (2 * scale) # x location of the arrow
        yloc_a = scale * xloc_a # y location of the arrow
    else:
        xloc_a = ax.get_xlim()[1] / 2 # x location of the arrow
        yloc_a = scale * xloc_a # y location of the arrow

    ax.plot(x, scale * x, color='r', linewidth=1)
    ax.annotate(f"scaling: {scale:.2f}", xy=(xloc_a, yloc_a), xycoords='data',
                xytext=(0.75, 0.9), textcoords='axes fraction', color='r',
                arrowprops=dict(arrowstyle="->", facecolor='r', edgecolor='r'),
                horizontalalignment='right', verticalalignment='top')

    fig.tight_layout()

    return ax

def nnp_evaluation_plots(Y_true, Y_pred, train_loss, pred_train_loss, valid_loss, loss_function, labels=None):
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:])

    ax1.set_title("SDR Embedding")
    plot_projection(Y_true, labels=labels, ax=ax1)

    ax2.set_title("SDR-NNP Embedding, test_loss = {:.6f}".format(loss_function(Y_true, Y_pred)))
    plot_projection(Y_pred, labels=labels, ax=ax2)

    ax3.set_title("Training History")
    ax3.plot(range(1,len(train_loss) + 1), train_loss, c="blue", label="Training loss")
    ax3.plot(range(1,len(train_loss) + 1), valid_loss, c="red", label="Validation loss")
    ax3.plot(range(1,len(train_loss) + 1), pred_train_loss, c="green", label="Training loss (inference)")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    ax3.set_yscale("log")

    h, l = ax3.get_legend_handles_labels()
    l = [": ".join([label, "final = {:.6f}, min = {:.6f}".format(loss[-1], np.min(loss))]) for label, loss in zip(l, [train_loss, valid_loss, pred_train_loss])]

    ax3.legend(handles=h, labels=l)
    ax3.grid()

    fig.tight_layout()

    return fig

class CustomConfusionMatrixDisplay(ConfusionMatrixDisplay):
    @classmethod
    def from_predictions_with_counts(
        cls,
        y_true,
        y_pred,
        *,
        labels=None,
        sample_weight=None,
        normalize=None,
        display_labels=None,
        include_values=True,
        xticks_rotation="horizontal",
        values_format=None,
        cmap="viridis",
        ax=None,
        colorbar=True,
        im_kw=None
    ):
        check_matplotlib_support(f"{cls.__name__}.from_predictions_with_counts")

        disp = cls.from_predictions(
            y_true, 
            y_pred, 
            labels=labels,
            sample_weight=sample_weight,
            normalize=normalize,
            display_labels=display_labels, 
            include_values=include_values,
            xticks_rotation=xticks_rotation,
            values_format=values_format,
            cmap=cmap,
            ax=ax,
            colorbar=colorbar,
            im_kw=im_kw
            )

        if normalize is not None:
            n_classes = disp.text_.shape[0]

            cm = confusion_matrix(y_true,
                                  y_pred,
                                  sample_weight=sample_weight,
                                  labels=labels,
                                  normalize=None)
            
            for i, j in product(range(n_classes), range(n_classes)):
                text_cm = format(cm[i, j], ".2g")
                text_d = format(cm[i, j], "d")
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                disp.text_[i, j].set_text(disp.text_[i, j].get_text() + "\n" + text_cm)

        return disp

# TODO:
# average local error plot