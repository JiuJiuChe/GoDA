import numpy as np

def pad_image(img, pad, mode='constant', pad_val=0):
    if isinstance(pad, int):
        pad = [pad, pad, pad, pad]  # order: top, bottom, left, right
    elif isinstance(pad, tuple) or isinstance(pad, list):
        assert len(pad) == 2 or len(pad) == 4
        if len(pad) == 2:
            pad = [pad[0], pad[0], pad[1], pad[1]]
    else:
        raise ValueError('padding can only be an integer or list with size of 2 or 4')
    
    # create a new imgage with padded size
    if len(img.shape) == 2:
        new_size = (img.shape[0]+pad[0]+pad[1], img.shape[1]+pad[2]+pad[3])
        new_image = np.zeros(new_size, dtype=img.dtype)
        # make paddings
        pad_plane(img, new_image, pad, mode, pad_val)
    elif len(img.shape) == 3:
        new_size = (img.shape[0]+pad[0]+pad[1], img.shape[1]+pad[2]+pad[3], img.shape[2])
        new_image = np.zeros(new_size, dtype=img.dtype)
        for i in range(img.shape[2]):
            pad_plane(img[:, :, i], new_image[:, :, i], pad, mode, pad_val)
    else:
        raise NotImplementedError(f'Not support data with shape {img.shape}')
    return new_image


def pad_plane(plane, pad_plane, pad, mode='constant', pad_val=0):
    pad_plane[pad[0]:-pad[1], pad[2]:-pad[3]] = plane
    if mode == 'constant':
        pad_plane[:pad[0], :] = pad_val
        pad_plane[-pad[1]:, :] = pad_val
        pad_plane[pad[0]:pad[1], :pad[2]] = pad_val
        pad_plane[pad[0]:pad[1], -pad[3]:] = pad_val
    else:
        raise NotImplementedError(f'Padding mode {mode} is not supported yet')


if __name__ == '__main__':
    from drill_bits.io import io_utils
    import matplotlib.pyplot as plt

    img = io_utils.omni_load(r'./src/test.png')
    padded = pad_image(img, [100, 10, 50, 200])

    plt.imshow(padded)
    plt.show()
