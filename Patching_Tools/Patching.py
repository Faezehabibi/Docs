import numpy as np
import jax.numpy as jnp



class create_patches:
    def __init__(self, img, patch, overlap):
        self.img = img
        self.patch = patch
        self.overlap = overlap
        self.height_patch, self.width_patch = patch
        self.height_over, self.width_over= overlap

        self.height, self.width = self.img.shape

        self.nw_patches = (self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        self.nh_patches = (self.height + self.height_over) // (self.height_patch - self.height_over) - 2

    def _add_frame(self):
        self.img = np.hstack((jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over))),
                         self.img,
                         jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over)))))
        self.img = np.vstack((jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1])),
                         self.img,
                         jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1]))))

        self.height, self.width = self.img.shape

        self.nw_patches = (self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        self.nh_patches = (self.height + self.height_over) // (self.height_patch - self.height_over) - 2

        # return self.img

    def create_patch(self, add_frame=False):
        if add_frame==True:
            self._add_frame()
        result = []
        for nh_ in range(self.nh_patches):
            for nw_ in range(self.nw_patches):
                img_ = self.img[(self.height_patch - self.height_over) * nh_: nh_ * (self.height_patch - self.height_over) + self.height_patch
                , (self.width_patch - self.width_over) * nw_: nw_ * (self.width_patch - self.width_over) + self.width_patch]
                result.append(img_)


        return jnp.array(result)