from torchvision.transforms.functional import pad
import numpy as np

class SquarePad:
    def __init__(self, global_max_wh=None):
        self.global_max_wh = global_max_wh
        
            
    def __call__(self, image):
        
        w, h = image.size
        
        if self.global_max_wh is not None:
            self.max_wh = self.global_max_wh
        else:           
            self.max_wh = np.max([w, h])
            
        h_padding = int((self.max_wh - w) / 2)
        v_padding = int((self.max_wh - h) / 2)
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
        return pad(image, padding, 0, 'constant')
    