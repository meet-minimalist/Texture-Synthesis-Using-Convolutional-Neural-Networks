from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

def resize_and_rescale_img(image_path, w, h, output_path_, output_filename):
    # This will resize the image to width x height dimensions and then scale down in the range of [0-1]   
    if os.path.isfile(image_path):
        img = Image.open(image_path)
        img_resized = img.resize(size=(w, h))
        if not os.path.exists(output_path_):
            os.makedirs(output_path_)
        img_resized.save(output_path_ + output_filename)
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = np.expand_dims(img_array, 0)
        img_array = img_array / 255
        
        return img_array
    else:
        print("No image found in given location.")

def post_process_and_display(cnn_output, output_path, output_filename, save_file=True):
    # This will take input_noise of (1, w, h, channels) shapped array taken from tensorflow operation
    # and ultimately displays the image
    
    x = np.squeeze(cnn_output)
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x))

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    img = Image.fromarray(x, mode='RGB')
    img.show()
    if save_file:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        img.save(output_path + output_filename)
    
    return x