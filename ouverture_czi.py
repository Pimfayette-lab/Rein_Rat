from czifile import imread
import napari
import numpy as np

img = np.squeeze(imread(r"C:\Users\pimfa\Documents\MAIA\Rein_de_rat\transfer_12956536_files_7b6ebfa1\2021_05_18__0973gtAQP2nov_rbAE1.czi"))

viewer = napari.Viewer()

colors = [
    "white",
    "green",
    "red",
    "blue"
    
]

for c in range(img.shape[0]):

    viewer.add_image(
        img[c],
        name=f"Canal {c}",
        blending="additive",
        colormap=colors[c % len(colors)]
    )

napari.run()