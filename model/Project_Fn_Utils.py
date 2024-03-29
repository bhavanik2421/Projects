
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from pathlib import Path
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder

def _get_root():
    return Path(__file__).parent

def load_data_1(path: Union[Path, str] = None, num_rows_to_display: int = 5):
    data = []

    # If path is not provided, set the default path
    if path is None:
        path = _get_root() / "data" / "dti"

    # Iterate over files in the specified directory
    for fname in (path).iterdir():
        # Check if the file has a .csv extension
        if fname.suffix == ".csv":
            # Read the CSV file into a Pandas DataFrame
            df = pd.read_csv(fname, header=None)

            # Extract the values as a NumPy array
            array_data = df.values

            # Append the array to the data list
            data.append(array_data)

            # Display the first few rows of the array
            print(f"Data from {fname}:")
            print(array_data[:num_rows_to_display])
            print("\n" + "="*30 + "\n")

    return data

def load_data_2(path: Union[Path, str] = None):
    entire_data = []
    if path is None:
        path = _get_root() / "data" / "recon_classification"
    
    for data_split in ["train", "val"]:  
        images, labels = [], []
        for root, dir, files in os.walk(path/data_split):
            root = Path(root)
            for file in files:
                if file.endswith('.png'):                
                    image = cv2.imread(str(root/file))
                    image = rgb2gray(image)
                    [nR, nC] = image.shape
                    image = image.reshape(1, nR, nC)
                    label = root.stem
                    images.append(image)
                    labels.append(label)
        entire_data.append(images)
        entire_data.append(labels)

    x_train, y_train, x_val, y_val = entire_data

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_val = le.fit_transform(y_val)

    return np.array(x_train), y_train, np.array(x_val), y_val

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def plot(
        vectors: list,
        labels: Optional[list] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        smoothing: Optional[int] = 1,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        root: Optional[Path] = None,
        filename: Optional[str] = None,
        
    ):
    """
    This function plots multiple vectors in a single figure.
    Args:
        vectors:            list of images to display
        labels:             list of labels for each image, optional
        xlabel:             label for x-axis, optional
        ylabel:             label for y-axis, optional
        title:              title for the figure, optional
        smoothing:          smoothing factor for the plot, optional
        xlim:               x-axis limits, optional
        ylim:               y-axis limits, optional
        root:               Root path to save, optional
        filename:           name of the file to save the figure, optional        
        
    """
    if not isinstance(vectors, list):
        vectors = [vectors]

    f, a = plt.subplots(1)
    for vector, label in zip(vectors, labels):
        a.plot(np.convolve(vector, np.ones((smoothing,)) / smoothing, mode='valid'), label=label)

    if xlabel:
        a.set_xlabel(xlabel)
    if ylabel:
        a.set_ylabel(ylabel)
    if title:
        a.set_title(title)
    if xlim:
        a.set_xlim(xlim)
    if ylim:
        a.set_ylim(ylim)
    a.legend()

    if root is None:
        root = _get_root()
    if isinstance(root, str):
        root = Path(root)
    root = root / "Results"
    if not root.exists() and filename:
        root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        plt.show()
    else:
        plt.savefig(root/f"{filename}_{title}", bbox_inches="tight", pad_inches=0.2)
    plt.close()

   
