#!/usr/bin/python
# -*- encoding: utf-8 -*-


import random
import math

import numpy as np
import cv2
import torch

class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            # img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0 ,1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, im_lb_dep_mask):
        """Call function to perform photometric distortion on images.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys like 'im', 'lb', and optionally 'dep', 'mask', 'specified_scale'.

        Returns:
            dict: A dictionary with the same keys as input, but with 'im' modified.
        """
        # Clone the input dictionary to avoid modifying the original
        result = {key: value for key, value in im_lb_dep_mask.items()}

        # Apply photometric distortion to the 'im' key
        result['im'] = self.brightness(result['im'])

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            result['im'] = self.contrast(result['im'])

        # random saturation
        result['im'] = self.saturation(result['im'])

        # random hue
        result['im'] = self.hue(result['im'])

        # random contrast
        if mode == 0:
            result['im'] = self.contrast(result['im'])
        
        # Preserve all other keys (specified_scale, crop_bbox, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]

        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb_dep_mask):
        """
        Call function to perform random resized crop on images, labels, and depth maps.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.

        Returns:
            dict: A dictionary with the same keys as input, but with cropped and resized values.
        """
        if self.size is None:
            return im_lb_dep_mask

        # Extract image, label, and depth map from the input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)  # Depth map is optional
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        specified_scale = im_lb_dep_mask.get('specified_scale', None)

        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        if mask is not None:
            assert im.shape[:2] == mask.shape[:2], "Image and mask must have the same spatial dimensions."
        
        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]

        # Resize image, label, and depth map
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        if dep is not None:
            dep = cv2.resize(dep, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        # If the resized dimensions match the crop size, return directly
        if (im_h, im_w) == (crop_h, crop_w):
            result = dict(im=im, lb=lb)
            if dep is not None:
                result['dep'] = dep
            if mask is not None:
                result['mask'] = mask
            # Preserve specified_scale if it exists
            if specified_scale is not None:
                result['specified_scale'] = specified_scale
            return result

        # Padding if necessary
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
            if dep is not None:
                dep = np.pad(dep, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
            if mask is not None:
                mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Random crop
        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        cropped_im = im[sh:sh + crop_h, sw:sw + crop_w, :].copy()
        cropped_lb = lb[sh:sh + crop_h, sw:sw + crop_w].copy()
        cropped_dep = dep[sh:sh + crop_h, sw:sw + crop_w].copy() if dep is not None else None
        cropped_mask = mask[sh:sh + crop_h, sw:sw + crop_w].copy() if mask is not None else None

        # Return the processed dictionary
        result = dict(im=cropped_im, lb=cropped_lb)
        if cropped_dep is not None:
            result['dep'] = cropped_dep
        if cropped_mask is not None:
            result['mask'] = cropped_mask
        # Preserve specified_scale if it exists
        if specified_scale is not None:
            result['specified_scale'] = specified_scale
        return result
        
class RandomResizCrop(object):
    '''
    size should be a tuple of (H, W)
    '''
    def __init__(self, resize_shape=(720, 1280), crop_size=(384, 384)):
        self.resize_shape = resize_shape
        self.crop_size = crop_size

    def __call__(self, im_lb_dep_mask):
        """
        Call function to perform random resize and crop on images, labels, and depth maps.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.

        Returns:
            dict: A dictionary with the same keys as input, but with resized and cropped values.
        """
        if self.crop_size is None:
            return im_lb_dep_mask

        # Extract image, label, and depth map from the input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)  # Depth map is optional
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        specified_scale = im_lb_dep_mask.get('specified_scale', None)

        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        if mask is not None:
            assert im.shape[:2] == mask.shape[:2], "Image and mask must have the same spatial dimensions."

        crop_h, crop_w = self.crop_size
        im_h, im_w = self.resize_shape
        
        # Resize image, label, and depth map
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        if dep is not None:
            dep = cv2.resize(dep, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        # If the resized dimensions match the crop size, return directly
        if (im_h, im_w) == (crop_h, crop_w):
            result = dict(im=im, lb=lb)
            if dep is not None:
                result['dep'] = dep
            if mask is not None:
                result['mask'] = mask
            return result

        # Padding if necessary
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=255)
            if dep is not None:
                dep = np.pad(dep, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
            if mask is not None:
                mask = np.pad(mask, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

        # Random crop
        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        
        # Create result dictionary
        result = dict(
            im=im[sh:sh+crop_h, sw:sw+crop_w, :].copy(),
            lb=lb[sh:sh+crop_h, sw:sw+crop_w].copy()
        )
        
        # Add depth map to result if it exists
        if dep is not None:
            result['dep'] = dep[sh:sh+crop_h, sw:sw+crop_w].copy()
        
        # Add mask to result if it exists
        if mask is not None:
            result['mask'] = mask[sh:sh+crop_h, sw:sw+crop_w].copy()
            
        # Preserve specified_scale if it exists
        if specified_scale is not None:
            result['specified_scale'] = specified_scale
            
        return result

class Resize(object):
    """Resize images, labels, and optionally depth maps.
    
    Args:
        resize_shape (tuple or int): Desired output size.
            - If tuple (h, w): target height and width
            - If int: target size for shorter edge when keep_ratio=True
        keep_ratio (bool): Whether to keep aspect ratio. Default: False.
            - If False: resize to exact resize_shape, may distort image
            - If True: resize keeping aspect ratio, output size may differ from resize_shape
        interpolation (dict, optional): Interpolation methods for different data types.
    """
    def __init__(self, resize_shape=(720, 1280), keep_ratio=False, interpolation=None):
        self.resize_shape = resize_shape
        self.keep_ratio = keep_ratio
        
        # Default interpolation methods
        default_interpolation = {
            'im': cv2.INTER_LINEAR,
            'lb': cv2.INTER_NEAREST,
            'dep': cv2.INTER_NEAREST,
            'mask': cv2.INTER_NEAREST
        }
        
        if interpolation is None:
            self.interpolation = default_interpolation
        else:
            self.interpolation = {**default_interpolation, **interpolation}
    
    def _calculate_resize_shape(self, orig_h, orig_w):
        """Calculate the actual resize shape based on keep_ratio setting.
        
        Args:
            orig_h (int): Original image height
            orig_w (int): Original image width
            
        Returns:
            tuple: (new_h, new_w, scale_factor)
        """
        if not self.keep_ratio:
            # Direct resize without keeping ratio
            if isinstance(self.resize_shape, int):
                # If single int provided, use it for both dimensions
                new_h = new_w = self.resize_shape
            else:
                new_h, new_w = self.resize_shape
            
            scale_h = new_h / orig_h
            scale_w = new_w / orig_w
            scale_factor = (scale_h, scale_w)
            
            return new_h, new_w, scale_factor
        
        else:
            # Keep aspect ratio
            if isinstance(self.resize_shape, int):
                # Single int: resize shorter edge to this size
                if orig_h < orig_w:
                    new_h = self.resize_shape
                    new_w = int(orig_w * (self.resize_shape / orig_h))
                else:
                    new_w = self.resize_shape
                    new_h = int(orig_h * (self.resize_shape / orig_w))
                scale_factor = new_h / orig_h

            elif isinstance(self.resize_shape, (tuple, list)):
                # Tuple or list (h, w): calculate scale factor to fit within bounds
                max_long_edge = max(self.resize_shape)
                max_short_edge = min(self.resize_shape)
                
                # Calculate scale factor that fits image within the bounds
                scale_factor = min(
                    max_long_edge / max(orig_h, orig_w),
                    max_short_edge / min(orig_h, orig_w)
                )
                
                # Calculate new dimensions
                new_h = int(orig_h * scale_factor)
                new_w = int(orig_w * scale_factor)
            
            else:
                raise TypeError(f"resize_shape must be int or tuple or list, got {type(self.resize_shape)}")
            
            return new_h, new_w, scale_factor
    
    def __call__(self, im_lb_dep_mask):
        """
        Resize images, labels, and optionally depth maps.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.
                - 'specified_scale': Pre-specified scale factor [optional]

        Returns:
            dict: A dictionary with the same keys as input, but with resized values.
        """
        # Extract image and label from the input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        # Extract depth map if available (optional)
        dep = im_lb_dep_mask.get('dep', None)
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        specified_scale = im_lb_dep_mask.get('specified_scale', None)

        # Get original dimensions
        orig_h, orig_w = im.shape[:2]
        
        # Calculate target dimensions
        new_h, new_w, scale_factor = self._calculate_resize_shape(orig_h, orig_w)
        # print(f"original size: ({orig_h}, {orig_w}), new size: ({new_h}, {new_w}), scale_factor: {scale_factor}")
        
        # If resize_shape is None or already matches the image dimensions, return early
        if self.resize_shape is None or (orig_h, orig_w) == (new_h, new_w):
            return im_lb_dep_mask
        
        # Resize image and label
        im = cv2.resize(im, (new_w, new_h), interpolation=self.interpolation['im'])
        lb = cv2.resize(lb, (new_w, new_h), interpolation=self.interpolation['lb'])
        
        # Create result dictionary
        result = dict(im=im, lb=lb)
        
        # Resize depth map if available
        if dep is not None:
            dep = cv2.resize(dep, (new_w, new_h), interpolation=self.interpolation['dep'])
            result['dep'] = dep
            
        # Resize mask if available
        if mask is not None:
            mask = cv2.resize(mask, (new_w, new_h), interpolation=self.interpolation['mask'])
            result['mask'] = mask
            
        # Add scale_factor to result for potential downstream use
        result['scale_factor'] = scale_factor
        
        # Preserve all other keys (specified_scale, crop_bbox, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]
            
        return result
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"resize_shape={self.resize_shape}, "
                f"keep_ratio={self.keep_ratio}, "
                f"interpolation={self.interpolation})")

class RandomResize(object):
    """
    Randomly resize images, labels, and optionally depth maps and masks.
    Supports both random sampling and specified scale usage.
    
    Args:
        scales (list or tuple): List of scale factors to randomly choose from.
        interpolation (dict, optional): Interpolation methods for different data types.
        return_scale (bool): Whether to return the used scale factor. Default: False.
    """
    
    def __init__(self, scales, max_size=None, interpolation=None, return_scale=True):
        if isinstance(scales, (int, float)):
            scales = [scales]
        self.scales = list(scales)
        self.max_size = max_size
        self.return_scale = return_scale
        
        # Default interpolation methods
        default_interpolation = {
            'im': cv2.INTER_LINEAR,
            'lb': cv2.INTER_NEAREST,
            'dep': cv2.INTER_NEAREST,
            'mask': cv2.INTER_NEAREST
        }
        
        if interpolation is None:
            self.interpolation = default_interpolation
        else:
            self.interpolation = {**default_interpolation, **interpolation}
    
    def __call__(self, im_lb_dep_mask):
        """
        Randomly resize or resize with specified scale.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional]
                - 'specified_scale': Specific scale to use [optional]

        Returns:
            dict: Always returns the transformed dictionary with optional 'specified_scale' key
        """
        # Extract data from input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)
        mask = im_lb_dep_mask.get('mask', None)
        specified_scale = im_lb_dep_mask.get('specified_scale', None)
        
        # Ensure consistent dimensions
        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        if mask is not None:
            assert im.shape[:2] == mask.shape[:2], "Image and mask must have the same spatial dimensions."
        
        # Determine scale factor
        if specified_scale is not None:
            scale = specified_scale
        else:
            scale = np.random.choice(self.scales)
            # print(f'scale: {scale}')
        
        # Get original dimensions
        orig_h, orig_w = im.shape[:2]
        
        # Apply max_size constraint if provided
        if self.max_size is not None:
            # Calculate the scale factor that would fit the longer edge to max_size
            max_scale = self.max_size / max(orig_h, orig_w)
            # Use the smaller of the two scales to ensure the constraint is met
            scale = min(scale, max_scale)
        
        # Calculate new dimensions
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Resize image
        resized_im = cv2.resize(im, (new_w, new_h), interpolation=self.interpolation['im'])
        
        # Resize label
        resized_lb = cv2.resize(lb, (new_w, new_h), interpolation=self.interpolation['lb'])
        
        # Create result dictionary
        result = dict(im=resized_im, lb=resized_lb)
        
        # Resize depth map if available
        if dep is not None:
            resized_dep = cv2.resize(dep, (new_w, new_h), interpolation=self.interpolation['dep'])
            result['dep'] = resized_dep
            
        # Resize mask if available
        if mask is not None:
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=self.interpolation['mask'])
            result['mask'] = resized_mask
        
        # Add used scale if return_scale is True
        if self.return_scale:
            result['specified_scale'] = scale
            
        # Preserve all other keys (specified_scale, crop_bbox, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]
        
        return result
    
    def __repr__(self):
        return f"{self.__class__.__name__}(scales={self.scales}, max_size={self.max_size}, interpolation={self.interpolation}, return_scale={self.return_scale})"

class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
        ignore_index (int): The label index to be ignored.
        pad_val (dict): Padding values for different data types.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255, 
                 pad_val=None):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index
        
        # Default padding values
        default_pad_val = {
            'im': (123, 116, 103),      # Black padding for images
            'gray_im': 0,  # Black padding for grayscale images
            'lb': 255,    # Ignore index for labels
            'dep': 0,     # Zero padding for depth maps
            'mask': 0     # Zero padding for masks
        }
        
        if pad_val is None:
            self.pad_val = default_pad_val
        else:
            self.pad_val = {**default_pad_val, **pad_val}

    def pad_to_crop_size(self, data_dict):
        """Pad all data to at least crop_size if necessary."""
        im = data_dict['im']
        lb = data_dict['lb']
        dep = data_dict.get('dep', None)
        mask = data_dict.get('mask', None)
        
        img_h, img_w = im.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # Calculate padding needed
        pad_h = max(crop_h - img_h, 0)
        pad_w = max(crop_w - img_w, 0)
        
        if pad_h == 0 and pad_w == 0:
            return data_dict  # No padding needed
        
        # using the top, bottom, left, right padding strategy
        # pad_top = pad_h // 2
        # pad_bottom = pad_h - pad_top
        # pad_left = pad_w // 2
        # pad_right = pad_w - pad_left
        
        # using the bottom and right padding strategy
        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w
        
        # Pad image
        if len(im.shape) == 3:  # Color image
            padded_im = np.zeros((img_h + pad_h, img_w + pad_w, im.shape[2]), dtype=im.dtype)
            for i, v in enumerate(self.pad_val['im']):
                padded_im[:, :, i] = v
            padded_im[pad_top:pad_top + img_h, pad_left:pad_left + img_w, :] = im
            # padded_im = np.pad(im, 
            #                  ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
            #                  mode='constant', 
            #                  constant_values=self.pad_val['im'])
        else:  # Grayscale image
            padded_im = np.pad(im, 
                             ((pad_top, pad_bottom), (pad_left, pad_right)), 
                             mode='constant', 
                             constant_values=self.pad_val['gray_im'])
        
        # Pad label
        padded_lb = np.pad(lb, 
                          ((pad_top, pad_bottom), (pad_left, pad_right)), 
                          mode='constant', 
                          constant_values=self.pad_val['lb'])
        
        # Create result dictionary
        result = {'im': padded_im, 'lb': padded_lb}
        
        # Pad depth map if available
        if dep is not None:
            padded_dep = np.pad(dep, 
                               ((pad_top, pad_bottom), (pad_left, pad_right)), 
                               mode='constant', 
                               constant_values=self.pad_val['dep'])
            result['dep'] = padded_dep
            
        # Pad mask if available
        if mask is not None:
            padded_mask = np.pad(mask, 
                                ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                mode='constant', 
                                constant_values=self.pad_val['mask'])
            result['mask'] = padded_mask
            
        return result

    def get_crop_bbox(self, img, crop_bbox=None):
        """Get a crop bounding box. Use provided bbox if available, otherwise random."""
        if crop_bbox is not None:
            # 使用指定的crop bbox（用于弱增强和强增强的空间对齐）
            return crop_bbox
        
        # 随机生成crop bbox
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img
    
    def crop_lb(self, lb, crop_bbox):
        """Crop from ``lb``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        lb = lb[crop_y1:crop_y2, crop_x1:crop_x2]
        return lb

    def __call__(self, im_lb_dep_mask):
        """Call function to randomly crop images, semantic segmentation maps, and depth maps.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.

        Returns:
            dict: Randomly cropped results with the same keys as input.
        """
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)
        mask = im_lb_dep_mask.get('mask', None)
        specified_scale = im_lb_dep_mask.get('specified_scale', None)
        specified_crop_bbox = im_lb_dep_mask.get('crop_bbox', None)  # 获取指定的crop bbox
        
        # Ensure consistent dimensions
        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        if mask is not None:
            assert im.shape[:2] == mask.shape[:2], "Image and mask must have the same spatial dimensions."
        
        # Step 1: Pad to crop size if necessary
        # If the image is smaller than crop size, pad it
        if im.shape[0] < self.crop_size[0] or im.shape[1] < self.crop_size[1]:
            padded_data = self.pad_to_crop_size(im_lb_dep_mask)
            
            # Extract padded data
            im = padded_data['im']
            lb = padded_data['lb']
            dep = padded_data.get('dep', None)
            mask = padded_data.get('mask', None)
        
        # Step 2: Get crop bounding box and apply category ratio constraint
        crop_bbox = self.get_crop_bbox(im, specified_crop_bbox)
        
        # 如果指定了crop_bbox，则跳过cat_max_ratio约束（用于弱增强和强增强的空间对齐）
        if specified_crop_bbox is None and self.cat_max_ratio < 1.:
            # Repeat 10 times to find a good crop
            for _ in range(10):
                seg_temp = self.crop_lb(lb, crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                    break
                else:
                    crop_bbox = self.get_crop_bbox(im)

        # Step 3: Apply cropping
        cropped_im = self.crop(im, crop_bbox)
        cropped_lb = self.crop_lb(lb, crop_bbox)
        
        # Create result dictionary
        result = dict(im=cropped_im, lb=cropped_lb)
        
        # Crop depth map if available
        if dep is not None:
            cropped_dep = self.crop_lb(dep, crop_bbox)
            result['dep'] = cropped_dep
            
        # Crop mask if available
        if mask is not None:
            cropped_mask = self.crop_lb(mask, crop_bbox)
            result['mask'] = cropped_mask
        
        # 保存crop_bbox用于弱增强和强增强的空间对齐
        result['crop_bbox'] = crop_bbox
        
        # Preserve all other keys (specified_scale, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]
        
        return result

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"crop_size={self.crop_size}, "
                f"cat_max_ratio={self.cat_max_ratio}, "
                f"ignore_index={self.ignore_index}, "
                f"pad_val={self.pad_val})")
    
class RandomRotate(object):
    '''
    Randomly rotate an image, its corresponding semantic label, and optionally a depth map.
    The image will be filled with (0,0,0) in empty areas, the label will be filled with 255,
    and the depth map (if provided) will be filled with 0.

    Parameters:
    angle (int or float): Maximum rotation angle in degrees. Rotation will be randomly chosen
                          between -angle and +angle.
    '''
    def __init__(self, angle=180, im_fill=(0, 0, 0), lb_fill=255, dep_fill=0, mask_fill=0):
        self.angle = angle
        self.im_fill = im_fill
        self.lb_fill = lb_fill
        self.dep_fill = dep_fill
        self.mask_fill = mask_fill  # Fill value for mask if provided
        self.p = 0.5
        
    def __call__(self, im_lb_dep_mask):
        if np.random.random() >= self.p:
            return im_lb_dep_mask
        
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)  # Depth map is optional
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        
        # Ensure the image and label have the same dimensions
        assert im.shape[:2] == lb.shape[:2]
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        if mask is not None:
            assert im.shape[:2] == mask.shape[:2], "Image and mask must have the same spatial dimensions."
        
        # Generate a random rotation angle between -self.angle and +self.angle
        angle = np.random.uniform(-self.angle, self.angle)
        
        # Get image dimensions
        h, w = im.shape[:2]
        
        # Calculate the rotation matrix
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        
        # Apply the affine transformation (rotation) with filling values
        im = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=self.im_fill)
        lb = cv2.warpAffine(lb, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.lb_fill)
        
        # Create result dictionary
        result = dict(im=im, lb=lb)
        
        # Rotate depth map if available
        if dep is not None:
            dep = cv2.warpAffine(dep, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.dep_fill)
            result['dep'] = dep
            
        # Rotate mask if available
        if mask is not None:
            mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.mask_fill)
            result['mask'] = mask    
        
        # Preserve all other keys (specified_scale, crop_bbox, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]
            
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(angle={self.angle}, im_fill={self.im_fill}, lb_fill={self.lb_fill}, dep_fill={self.dep_fill}, mask_fill={self.mask_fill}, p={self.p})"

class RandomHorizontalFlip(object):
    """
    Randomly flip the image, label, and optionally depth map horizontally.

    Args:
        p (float): Probability of performing a flip. Default: 0.5
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb_dep_mask):
        """
        Call function to randomly flip images, labels, and depth maps horizontally.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.
                - 'do_flip': Pre-specified flip decision (True/False) [optional]

        Returns:
            dict: Dictionary with the same keys as input, but with flipped values if random condition is met, plus 'did_flip'.
        """
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        dep = im_lb_dep_mask.get('dep', None)  # Depth map is optional
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        specified_scale = im_lb_dep_mask.get('specified_scale', None)
        crop_bbox = im_lb_dep_mask.get('crop_bbox', None)  # 传递crop_bbox
        
        # 随机决定是否flip (flip可以不一致，在模型预测时判断)
        do_flip = np.random.random() < self.p
        
        if not do_flip:
            # 不进行flip，直接返回
            result = im_lb_dep_mask.copy()
            result['did_flip'] = False
            return result
        
        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        
        # Create result dictionary with flipped image and label
        result = dict(
            im=im[:, ::-1, :].copy(),  # Flip horizontally
            lb=lb[:, ::-1].copy(),     # Flip horizontally
        )
        
        # Flip depth map if available
        if dep is not None:
            result['dep'] = dep[:, ::-1].copy()  # Flip horizontally
        
        if mask is not None:
            result['mask'] = mask[:, ::-1].copy()  # Flip horizontally
            
        if specified_scale is not None:
            result['specified_scale'] = specified_scale
        
        if crop_bbox is not None:
            result['crop_bbox'] = crop_bbox  # 传递crop_bbox
        
        # 记录进行了flip
        result['did_flip'] = True
        
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class RandomGrayscale(object):
    """
    Randomly convert image to grayscale with a given probability.
    
    Args:
        p (float): Probability of converting to grayscale. Default: 0.3
    """
    def __init__(self, p=0.3):
        self.p = p
    
    def __call__(self, im_lb_dep_mask):
        im = im_lb_dep_mask['im']
        if random.random() < self.p:
            # Convert to grayscale
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # Convert back to 3 channels
            im = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            im_lb_dep_mask['im'] = im
        return im_lb_dep_mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class GaussianBlur(object):
    """
    Apply Gaussian blur to the image.
    
    Args:
        kernel_size (int or tuple): Size of the Gaussian kernel. 
            If int, a square kernel is used. Default: (3, 3)
        sigma (tuple): Range of sigma values for Gaussian kernel.
            A random sigma is chosen uniformly from [sigma[0], sigma[1]].
            Default: (0.1, 2.0)
        p (float): Probability of applying blur. Default: 0.5
    """
    def __init__(self, kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
    
    def __call__(self, im_lb_dep_mask):
        im = im_lb_dep_mask['im']
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            im = cv2.GaussianBlur(im, self.kernel_size, sigma)
            im_lb_dep_mask['im'] = im
        return im_lb_dep_mask
    
    def __repr__(self):
        return f'{self.__class__.__name__}(kernel_size={self.kernel_size}, sigma={self.sigma}, p={self.p})'


class ColorJitter(object):
    """
    Apply color jittering to images (brightness, contrast, saturation, hue adjustments).
    
    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue (float): How much to jitter hue. hue_factor is chosen uniformly from 
            [-hue, hue], should be in [0, 0.5]. Hue is the color component in HSV space.
        p (float): Probability of applying the transform. Default: 1.0 (always apply).
    """
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None, p=1.0):
        if brightness is not None and brightness >= 0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        else:
            self.brightness = None
            
        if contrast is not None and contrast >= 0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        else:
            self.contrast = None
            
        if saturation is not None and saturation >= 0:
            self.saturation = [max(1-saturation, 0), 1+saturation]
        else:
            self.saturation = None
        
        if hue is not None:
            assert 0 <= hue <= 0.5, "Hue jitter value should be in [0, 0.5]"
            self.hue = [-hue, hue]
        else:
            self.hue = None
        
        self.p = p

    def __call__(self, im_lb_dep_mask):
        """
        Apply color jittering to images with probability p.
        
        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.
                
        Returns:
            dict: A dictionary with the same keys as input, but with color-jittered image.
        """
        # Extract image and label from the input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        # Extract depth map if available (optional)
        dep = im_lb_dep_mask.get('dep', None)
        # Extract mask if available (optional)
        mask = im_lb_dep_mask.get('mask', None)
        specified_scale = im_lb_dep_mask.get('specified_scale', None)
        
        assert im.shape[:2] == lb.shape[:2], "Image and label must have the same spatial dimensions."
        if dep is not None:
            assert im.shape[:2] == dep.shape[:2], "Image and depth map must have the same spatial dimensions."
        
        # Apply color jittering with probability p
        if random.random() < self.p:
            # Apply brightness jittering
            if self.brightness is not None:
                rate = np.random.uniform(*self.brightness)
                im = self.adj_brightness(im, rate)
                
            # Apply contrast jittering
            if self.contrast is not None:
                rate = np.random.uniform(*self.contrast)
                im = self.adj_contrast(im, rate)
                
            # Apply saturation jittering
            if self.saturation is not None:
                rate = np.random.uniform(*self.saturation)
                im = self.adj_saturation(im, rate)
            
            # Apply hue jittering
            if self.hue is not None:
                delta = np.random.uniform(*self.hue)
                im = self.adj_hue(im, delta)
        
        # Create result dictionary with modified image and original label
        result = dict(im=im, lb=lb)
        
        # Add depth map to result if it exists (color jittering doesn't affect depth)
        if dep is not None:
            result['dep'] = dep
            
        # Add mask to result if it exists (color jittering doesn't affect mask)
        if mask is not None:
            result['mask'] = mask
        
        # Preserve all other keys (specified_scale, crop_bbox, do_flip, etc.)
        for key in im_lb_dep_mask:
            if key not in result:
                result[key] = im_lb_dep_mask[key]
            
        return result

    def adj_saturation(self, im, rate):
        """Adjust image saturation."""
        M = np.float32([
            [1+2*rate, 1-rate, 1-rate],
            [1-rate, 1+2*rate, 1-rate],
            [1-rate, 1-rate, 1+2*rate]
        ])
        shape = im.shape
        im = np.matmul(im.reshape(-1, 3), M).reshape(shape)/3
        im = np.clip(im, 0, 255).astype(np.uint8)
        return im

    def adj_brightness(self, im, rate):
        """Adjust image brightness."""
        table = np.array([
            i * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]

    def adj_contrast(self, im, rate):
        """Adjust image contrast."""
        table = np.array([
            74 + (i - 74) * rate for i in range(256)
        ]).clip(0, 255).astype(np.uint8)
        return table[im]
    
    def adj_hue(self, im, delta):
        """Adjust image hue.
        
        Args:
            im (np.ndarray): BGR image
            delta (float): Hue offset in range [-0.5, 0.5], will be scaled to [-180, 180] degrees
        
        Returns:
            np.ndarray: BGR image with adjusted hue
        """
        # Convert BGR to HSV
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust hue channel (scale delta from [-0.5, 0.5] to [-180, 180] degrees)
        # In OpenCV, Hue is in range [0, 180] for uint8
        hsv[:, :, 0] = (hsv[:, :, 0] + delta * 180) % 180
        
        # Convert back to BGR
        hsv = hsv.astype(np.uint8)
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return im
        
    def __repr__(self):
        """Return a string representation of the ColorJitter object."""
        format_range = lambda r: f"[{r[0]:.2f}, {r[1]:.2f}]" if r else None
        return (f"{self.__class__.__name__}("
                f"brightness={format_range(self.brightness)}, "
                f"contrast={format_range(self.contrast)}, "
                f"saturation={format_range(self.saturation)}, "
                f"hue={format_range(self.hue)}, "
                f"p={self.p})")

class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        
    def imnormalize(self, img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        return self.imnormalize_(img, mean, std, to_rgb)
    
    def imnormalize_(self, img, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.

        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        return img

    def __call__(self, im_lb_dep_mask):
        """Call function to normalize images.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.

        Returns:
            dict: A dictionary with the same keys as input, but with normalized image.
        """
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        # Extract depth map if available (optional)
        dep = im_lb_dep_mask.get('dep', None)
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        
        # Normalize the image
        im = self.imnormalize(im, self.mean, self.std, self.to_rgb)
        
        # Create result dictionary
        result = dict(im=im, lb=lb)
        
        # Add depth map to result if it exists (normalization doesn't affect depth)
        if dep is not None:
            result['dep'] = dep
        
        # Add mask to result if it exists (normalization doesn't affect mask)
        if mask is not None:
            result['mask'] = mask
            
        return result

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

class ToTensor(object):
    '''
    Convert numpy arrays to torch tensors.
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = torch.as_tensor(mean)[:, None, None]
        self.std = torch.as_tensor(std)[:, None, None]

    def __call__(self, im_lb_dep_mask):
        """
        Convert numpy arrays to torch tensors.

        Args:
            im_lb_dep_mask (dict): A dictionary containing keys:
                - 'im': Image (H, W, C)
                - 'lb': Semantic segmentation label (H, W)
                - 'dep': Depth map (H, W) [optional]
                - 'mask': Mask (H, W) [optional], 0 for invalid pixels, 1 for valid pixels.

        Returns:
            dict: A dictionary with the same keys as input, but with tensor values.
        """
        # Extract image and label from the input dictionary
        im, lb = im_lb_dep_mask['im'], im_lb_dep_mask['lb']
        # Extract depth map if available (optional)
        dep = im_lb_dep_mask.get('dep', None)
        mask = im_lb_dep_mask.get('mask', None)  # Mask is optional
        
        # Process image: transpose, convert to tensor, normalize
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        im = im.sub_(self.mean.to(im.device)).div_(self.std.to(im.device)).clone()
        
        # Process label: convert to tensor if not None
        if lb is not None:
            lb = torch.from_numpy(lb.astype(np.int64).copy()).clone()
        
        # Create result dictionary with converted image and label
        result = dict(im=im, lb=lb)
        
        # Process depth map if available
        if dep is not None:
            # Convert depth map to tensor (keep as float32)
            # Note: we don't normalize depth as it represents actual distances
            dep = torch.from_numpy(dep.astype(np.float32).copy()).clone()
            result['dep'] = dep
            
        if mask is not None:
            # Convert mask to tensor (keep as uint8)
            mask = torch.from_numpy(mask.astype(np.uint8).copy()).clone()
            result['mask'] = mask
            
        return result

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    
class ToTensor_Img(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''
    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im):
        im = im.transpose(2, 0, 1).astype(np.float32)
        im = torch.from_numpy(im).div_(255)
        dtype, device = im.dtype, im.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)[:, None, None]
        std = torch.as_tensor(self.std, dtype=dtype, device=device)[:, None, None]
        im = im.sub_(mean).div_(std).clone()

        return im

class Compose(object):
    """
    Compose several transformations together.
    
    Args:
        do_list (list): List of transform objects to compose.
    """
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, im_lb_dep_mask):
        """
        Apply all transformations sequentially.

        Args:
            im_lb_dep_mask (dict): A dictionary containing transformation data and parameters

        Returns:
            dict: The transformed dictionary
        """
        if not self.do_list:
            return im_lb_dep_mask
        
        result = im_lb_dep_mask
        
        for comp in self.do_list:
            result = comp(result)
        
        return result
            
    def __repr__(self):
        """Return string representation of all composed transformations."""
        format_string = self.__class__.__name__ + '('
        for t in self.do_list:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string

class TransformationBase(object):
    def __init__(self, resize_shape, keep_ratio=False, scale=None, cropsize=None, cat_max_ratio=1.0, flip=0, photo_metric=False, rotate=0):
        trans_list = [Resize(resize_shape, keep_ratio=keep_ratio)]
        if scale is not None:
            trans_list.append(RandomResize(scale, return_scale=False))
        if cropsize is not None:
            trans_list.append(RandomCrop(cropsize, cat_max_ratio))
        if flip > 0:
            trans_list.append(RandomHorizontalFlip(flip))
        if rotate > 0:
            trans_list.append(RandomRotate(rotate))
        if photo_metric:
            trans_list.append(PhotoMetricDistortion())
        self.trans_func = Compose(trans_list)

    def prepare_rcs_retry(self, im_lb_dep_mask):
        """Cache the deterministic prefix used by repeated RCS crops.

        RCS retries only need to resample the stochastic crop/augmentation
        suffix.  A fixed ``Resize`` and a ``RandomResize`` with an explicitly
        supplied scale are deterministic, so running them once preserves the
        sampling distribution while avoiding repeated OpenCV resize work.
        """
        result = im_lb_dep_mask
        split_index = 0
        for transform in self.trans_func.do_list:
            cacheable = isinstance(transform, Resize)
            cacheable = cacheable or (
                isinstance(transform, RandomResize)
                and result.get('specified_scale') is not None
            )
            if not cacheable:
                break
            result = transform(result)
            split_index += 1
        return result, split_index

    def apply_rcs_retry(self, prepared, split_index):
        """Apply the uncached stochastic suffix for one RCS crop attempt."""
        result = prepared
        for transform in self.trans_func.do_list[split_index:]:
            result = transform(result)
        return result

    def __call__(self, im_lb_dep_mask, keep_original_label=False):
        """
        Apply transformations to the input data.
        
        Args:
            im_lb_dep_mask (dict): Input data dictionary, may contain 'specified_scale'
            keep_original_label (bool): Whether to keep the original label unchanged
            
        Returns:
            dict: Transformed data with optional 'specified_scale' key
        """
        if keep_original_label:
            ori_lb = im_lb_dep_mask['lb'].copy()
        
        # Apply transformations
        result = self.trans_func(im_lb_dep_mask)
        
        # Replace label with original if requested
        if keep_original_label:
            result['lb'] = ori_lb
        
        return result

class TransformationTrain(TransformationBase):
    def __call__(self, im_lb_dep_mask):
        """
        Apply training transformations.
        
        Args:
            im_lb_dep_mask (dict): Input data dictionary, may contain 'specified_scale'
            
        Returns:
            dict: Transformed data with optional 'specified_scale' key
        """
        return super().__call__(im_lb_dep_mask, keep_original_label=False)

class TransformationVal(TransformationBase):
    def __call__(self, im_lb_dep_mask):
        """
        Apply validation transformations (keeps original label).
        
        Args:
            im_lb_dep_mask (dict): Input data dictionary, may contain 'specified_scale'
            
        Returns:
            dict: Transformed data with optional 'specified_scale' key
        """
        return super().__call__(im_lb_dep_mask, keep_original_label=True)
    
class TransformationTest(TransformationBase):
    def __call__(self, im_lb_dep_mask):
        """
        Apply test transformations (keeps original label).
        
        Args:
            im_lb_dep_mask (dict): Input data dictionary, may contain 'specified_scale'
            
        Returns:
            dict: Transformed data with optional 'specified_scale' key
        """
        return super().__call__(im_lb_dep_mask, keep_original_label=True)

if __name__ == '__main__':
    gta_resize = [720, 1280]
    gta_scale = [x * 0.1 for x in range(5,21)]
    gta_cropsize = [512, 512]
    
    city_resize = [512, 1024]
    city_scale = [0.5, 0.75, 1.0, 1.25, 1.5]
    city_cropsize = [512, 512]
    
    trans_gta = TransformationTrain(
        resize_shape=gta_resize,
        scale=gta_scale,
        cropsize=gta_cropsize,
        photo_metric=True,
        cat_max_ratio=0.75,
        flip=0.5,
    )

    trans_city = TransformationTrain(
        resize_shape=city_resize,
        scale=city_scale,
        cropsize=city_cropsize,
        photo_metric=True,
        cat_max_ratio=0.75,
        flip=0.5,
    )
    
    gta_img = cv2.imread('images/gta.png')
    gta_lb = cv2.imread('images/gta_lb.png', cv2.IMREAD_GRAYSCALE)
    gta_data = {
        'im': gta_img,
        'lb': gta_lb,
        'dep': None,
        'mask': None
    }
    city_img = cv2.imread('images/city.png')
    city_lb = cv2.imread('images/city_lb.png', cv2.IMREAD_GRAYSCALE)
    city_data = {
        'im': city_img,
        'lb': city_lb,
        'dep': None,
        'mask': None
    }
    
    # 检查图像是否成功加载
    if gta_img is None or city_img is None:
        print("Warning: Could not load images. Creating dummy data for testing.")
        gta_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        gta_lb = np.random.randint(0, 19, (600, 800), dtype=np.uint8)
        city_img = np.random.randint(0, 255, (500, 900, 3), dtype=np.uint8)
        city_lb = np.random.randint(0, 19, (500, 900), dtype=np.uint8)
        
        gta_data = {'im': gta_img, 'lb': gta_lb, 'dep': None, 'mask': None}
        city_data = {'im': city_img, 'lb': city_lb, 'dep': None, 'mask': None}
    
    print(f"Original GTA data shape: im={gta_data['im'].shape}, lb={gta_data['lb'].shape}")
    print(f"Original City data shape: im={city_data['im'].shape}, lb={city_data['lb'].shape}")
    
    # 测试 GTA 变换
    scale = np.random.choice(gta_scale)
    gta_data['specified_scale'] = scale  # 模拟指定的缩放比例
    print(f"Applying GTA transformation with specified scale: {scale}")
    city_data['specified_scale'] = scale  # 同样的缩放比例应用于 City 数据
    
    gta_result = trans_gta(gta_data)
    city_result = trans_city(city_data)
        
    print(f"Final GTA result shape: im={gta_result['im'].shape}, lb={gta_result['lb'].shape}")
    print(f"Final City result shape: im={city_result['im'].shape}, lb={city_result['lb'].shape}")
    
    # 验证最终尺寸是否符合预期
    expected_shape = (512, 512)
    assert gta_result['im'].shape[:2] == expected_shape, f"GTA image shape {gta_result['im'].shape[:2]} != {expected_shape}"
    assert gta_result['lb'].shape == expected_shape, f"GTA label shape {gta_result['lb'].shape} != {expected_shape}"
    assert city_result['im'].shape[:2] == expected_shape, f"City image shape {city_result['im'].shape[:2]} != {expected_shape}"
    assert city_result['lb'].shape == expected_shape, f"City label shape {city_result['lb'].shape} != {expected_shape}"
    
    print("✅ All tests passed!")
    
    # 创建输出目录并保存图像
    import os
    os.makedirs('images', exist_ok=True)
    cv2.imwrite('images/transformed_gta_im.png', gta_result['im'])
    cv2.imwrite('images/transformed_gta_lb.png', gta_result['lb'])
    cv2.imwrite('images/transformed_city_im.png', city_result['im'])
    cv2.imwrite('images/transformed_city_lb.png', city_result['lb'])
    print("🖼️ Transformed images saved!")
