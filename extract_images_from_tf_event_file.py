import os
import glob
import numpy as np
from scipy import misc
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

EVAL_DIR = './tmp/20170906-17:26:07/eval'
IMAGE_SHAPE = [256, 256]

event_acc = EventAccumulator(EVAL_DIR)
event_acc.Reload()
image_tags = event_acc.Tags()['images']
total = len(image_tags)
for i, tag in enumerate(image_tags):
    encoded_image_string = event_acc.Images(tag)[0].encoded_image_string
    ty = tag.split('/')[0]
    index = tag.split('/')[2]
    output_dir = os.path.join(EVAL_DIR, ty)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open('%s/%s.png' % (output_dir, index), 'wb') as f:
        f.write(encoded_image_string)

    print '%d/%d exporting done.' % (i, total)


types = [name for name in os.listdir(EVAL_DIR) if os.path.isdir(os.path.join(EVAL_DIR, name))]

n_row = 8
n_col = 3

for ty in types:
    png_paths = glob.glob(os.path.join(EVAL_DIR, ty, '*.png'))
    imgs = []
    img_at_once = np.ones(((min(n_col, len(png_paths) / n_row + 1)*IMAGE_SHAPE[1]), n_row*IMAGE_SHAPE[0]),
                          dtype=np.uint8) * 255
    for i, path in enumerate(png_paths):
        if i == n_row * n_col:
            break
        im = misc.imread(path)
        r = i / n_row
        c = i % n_row
        img_at_once[r * IMAGE_SHAPE[0]: (r+1) * IMAGE_SHAPE[0], c * IMAGE_SHAPE[1]: (c+1) * IMAGE_SHAPE[1]] = im

    misc.imsave(os.path.join(EVAL_DIR, '%s_at_once.png' % ty), img_at_once.astype(np.uint8))
