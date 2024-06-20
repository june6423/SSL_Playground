import nvidia.dali.fn as fn
from nvidia.dali import pipeline, ops, types

import collections
import numpy as np
from random import shuffle
import glob, os


def create_dali_pipeline(batch_size, num_threads, device_id, images_dir, shuffle_files=True, n_view=2):
    class ExternalInputIterator(object):
        def __init__(self, batch_size, mode="train", shuffl=True):
            self.images_dir = images_dir
            self.batch_size = batch_size
            with open(os.path.join(self.images_dir, "file_list.txt")) as f:
                self.files = [line.rstrip() for line in f if line != ""]
            if shuffle_files:
                shuffle(self.files)

        def __iter__(self):
            self.i = 0
            self.n = len(self.files)
            return self
        
        def __len__(self):
            return len(self.files)

        def __next__(self):
            batch = []
            labels = []
            for _ in range(self.batch_size):
                jpeg_filename, label = self.files[self.i].split(" ")
                f = open(jpeg_filename, "rb")
                batch.append(np.frombuffer(f.read(), dtype=np.uint8))
                labels.append(np.array([label], dtype=np.uint8))
                self.i = (self.i + 1) % self.n
            return (batch, labels)
        
    def augment_images(images):
        images = fn.resize(images, resize_x=32, resize_y=32)
        # # Apply random crop
        # images = fn.random_resized_crop(images, size=[32, 32])
        # # Apply brightness-contrast enhancement
        # images = fn.brightness_contrast(images, contrast=fn.random.uniform(range=[0.5, 1.5]) * fn.random.coin_flip(probability=0.5))
        # # Apply random flip
        # images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5), vertical=False)
        # # Transpose the images to change the channel order
        images = fn.transpose(images, perm=[2, 0, 1])
        return images

    eii = ExternalInputIterator(batch_size)

    pipe = pipeline.Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        jpegs, labels = fn.external_source(source=eii, num_outputs=2, dtype=types.UINT8)
        decode = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        decode = fn.cast(decode, dtype=types.FLOAT)
        if n_view == 2:
            augmented1 = augment_images(decode)
            augmented2 = augment_images(decode)
            pipe.set_outputs(augmented1, augmented2, labels)
        else:
            augmented = augment_images(decode)
            pipe.set_outputs(augmented, labels)

    pipe.build()
    return eii, pipe


class DALIIterator:
    def __init__(self, pipeline, size, n_view=2):
        self.pipeline = pipeline
        self.size = size
        self.n_view = n_view

    def __iter__(self):
        self.i = 0
        self.n = self.size
        return self

    def __len__(self):
        return int(self.size)
    
    def __next__(self):
        if self.i >= self.n:
            raise StopIteration
        data = self.pipeline.run()
        self.i += 1
        # Convert the data to PyTorch tensors and reshape them
        if self.n_view == 2:
            images1, images2 , labels = data
            return images1, images2, labels
        else:
            images, labels = data
            return images, labels

    next = __next__


def main():
    batch_size = 256
    eii, pipe = create_dali_pipeline(batch_size=batch_size, num_threads=2, device_id=0, images_dir="/shared/data/dongjun.nam/CIFAR-10-images")
    dali_iterator = DALIIterator(pipe, size=int(len(eii) / batch_size))

    for i, data in enumerate(dali_iterator):
        images1,images2, labels = data

if __name__ == "__main__":
    main()
    # Needs Improvement!
    # Current Elapsed time per epoch
    # DALI : 77s
    # Naive Pytorch : 27s