import PIL.Image

path_to_image = './daeho.png'
rgba_image = PIL.Image.open(path_to_image)
rgb_image = rgba_image.convert('RGB')

rgb_image.save("dae2.png")
