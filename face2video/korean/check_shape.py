import numpy as np
from PIL import Image

path ='./dae2.png'
i = np.array(Image.open(path))
print(i.shape)
