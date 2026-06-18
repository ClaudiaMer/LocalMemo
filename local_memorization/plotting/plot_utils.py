import matplotlib.pyplot as plt
import numpy as np

def set_nice_params():
    global panel_label_counter 
    plt.rc('xtick', direction='out', color='gray', 
           labelsize=8)
    plt.rc('ytick', direction='out', color='gray',
           labelsize=8)
    plt.rc('patch', edgecolor="grey")
    font = {'size'   : 8}
    #plt.rc('font', **font)
    plt.rc('axes.spines', top=False, right=False)
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    #plt.rc('font', serif='Times New Roman') 
    panel_label_counter = 0
    

def get_next_panel_label():
    global panel_label_counter 
    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    panel_labels = [letter +") " for letter in panel_letters]
    label = panel_labels[panel_label_counter]
    panel_label_counter +=1
    return label

def reset_panel_label_counter():
    global panel_label_counter 
    panel_label_counter = 0


def show_images(images, title="", key=""):
    """Shows the provided images as sub-pictures in a square"""

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                #plt.imshow(images[idx].reshape(pixel, pixel, n_channels), cmap="gray")
                f = plt.imshow(images[idx], cmap="bone")
                plt.colorbar(f)
                plt.axis('off')
                idx += 1
    fig.suptitle(title, fontsize=30)
    
    # Showing the figure
    plt.tight_layout()
    plt.savefig("figures/samples"+key+".png")
    #plt.show()

def plot_curve(x,y,yerr, ax, label, color, NUM_BATCHES=10, linestyle="-"): 
    yerr = yerr
    ax.plot(x,y, label=label, color=color, linestyle=linestyle)
    ax.fill_between(x, y-yerr, y+yerr, color=color, alpha=0.3)