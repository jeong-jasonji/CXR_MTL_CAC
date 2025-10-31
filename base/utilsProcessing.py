from __future__ import print_function
import os
import numpy as np
from PIL import Image

### general pytorch utils from: https://github.com/NVIDIA/pix2pixHD/blob/master/util/util.py ###

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

### pydicom_PIL specific processing ###

class pydicom_PIL():
    """
    converting the previous pydicom_PIL.py into a class for use
    """
    def __init__(self):
        have_PIL = True
        try:
            import PIL.Image
        except ImportError:
            have_PIL = False

        have_numpy = True
        try:
            import numpy as np
        except ImportError:
            have_numpy = False


    def get_LUT_value(self, data, window, level):
        """Apply the RGB Look-Up Table for the given
           data and window/level value."""
        if not have_numpy:
            raise ImportError("Numpy is not available."
                              "See http://numpy.scipy.org/"
                              "to download and install")

        return np.piecewise(data,
                            [data <= (level - 0.5 - (window - 1) / 2),
                             data > (level - 0.5 + (window - 1) / 2)],
                            [0, 255, lambda data: ((data - (level - 0.5)) /
                                                   (window - 1) + 0.5) * (255 - 0)])

    def window_image(self, dcm, window_center, window_width):
        if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
            correct_dcm(dcm)

        img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        return img

    def window_MR_image(self, dcm, window_center, window_width):
        # because the HCC images don't have RescaleSlope or RescaleIntercept (mostly found on CT and PET where the units
        # like HU (-2000, 2000) are large and need to be rescaled.
        img = dcm.pixel_array
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        return img

    def correct_dcm(self, dcm):
        x = dcm.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    def get_PIL_image(self, dataset):
        """Get Image object from Python Imaging Library(PIL)"""
        if not have_PIL:
            raise ImportError("Python Imaging Library is not available. "
                              "See http://www.pythonware.com/products/pil/ "
                              "to download and install")

        if ('PixelData' not in dataset):
            raise TypeError("Cannot show image -- DICOM dataset does not have "
                            "pixel data")
        # can only apply LUT if these window info exists
        if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
            bits = dataset.BitsAllocated
            samples = dataset.SamplesPerPixel
            if bits == 8 and samples == 1:
                mode = "L"
            elif bits == 8 and samples == 3:
                mode = "RGB"
            elif bits == 16:
                # not sure about this -- PIL source says is 'experimental'
                # and no documentation. Also, should bytes swap depending
                # on endian of file and system??
                mode = "I;16"
            else:
                raise TypeError("Don't know PIL mode for %d BitsAllocated "
                                "and %d SamplesPerPixel" % (bits, samples))

            # PIL size = (width, height)
            size = (dataset.Columns, dataset.Rows)

            # Recommended to specify all details
            # by http://www.pythonware.com/library/pil/handbook/image.htm
            im = PIL.Image.frombuffer(mode, size, dataset.PixelData,
                                      "raw", mode, 0, 1)

        else:
            ew = dataset['WindowWidth']
            ec = dataset['WindowCenter']
            ww = int(ew.value[0] if ew.VM > 1 else ew.value)
            wc = int(ec.value[0] if ec.VM > 1 else ec.value)
            image = window_MR_image(dataset, 150, 300)
            # Convert mode to L since LUT has only 256 values:
            #   http://www.pythonware.com/library/pil/handbook/image.htm
            im = PIL.Image.fromarray(image).convert('L')

        return im

    def show_PIL(self, dataset):
        """Display an image using the Python Imaging Library (PIL)"""
        im = get_PIL_image(dataset)
        im.show()
