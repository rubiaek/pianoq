

def calc_contrast(im):
    return (im**2).mean() / im.mean()**2