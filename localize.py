import cv2
import numpy as np
from time import time

# from abc import ABC, abstractmethod

from find_peaks import *
from filtering import *
from fitting import fitCircleLS
from utils import sharpen, autoScaleABC

FIND_PUPIL = 0
FIND_IRIS = 1

POLAR_GRADIENT = 0
POLAR_CANNY = 1
POLAR_CURVEFIT = 2


# TODO abstract parent class
# class Localizer(ABC):
#     def __init__(self) -> None:
#         super().__init__()
        
#     @abstractmethod 
#     def find(self):
#         pass

def image_preprocessing(source):
    if len(source.shape) == 3:
        img = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    elif  len(source.shape) == 2:
        img = source.copy().astype(np.uint8)
    
    # INPAINTING DATA IN IMAGE
    inp_mask = np.zeros(img.shape, dtype=np.uint8)
    inp_mask[0, :19] = 1
        
    img = cv2.inpaint(img, inp_mask, 1, cv2.INPAINT_TELEA)
    
    # SHARPENING
    img = sharpen(img, 1, gauss_ksize=3, gauss_sigmaX=1)
    
    # BLURING
    morphBlurSize = 9

    operations_queue = [cv2.MORPH_OPEN]
    iterations_queue = [1]

    for operation, iterations in zip(operations_queue, iterations_queue):
        img = cv2.morphologyEx(img, operation, cv2.getStructuringElement(cv2.MORPH_CROSS, (morphBlurSize, morphBlurSize)), iterations=iterations)
    
    img = cv2.medianBlur(img, 7)

    img = autoScaleABC(img)
    return img

class PolarFinder():
    def __init__(self, mode: int, center_estimation: tuple[int],
                 angles: list[int] = None,
                 peaks_find_method: int = POLAR_GRADIENT,
                 filter_model: FilterModel = None,
                 fitting_method: callable = fitCircleLS) -> None:
        
        assert mode in [FIND_PUPIL, FIND_IRIS], "Wrong mode value. Use one of: irisloc.FIND_PUPIL, irisloc.FIND_IRIS"
        self.mode = mode
        self.center_estimation = center_estimation
        self.peaks_find_method = peaks_find_method
        if angles == None:
            if mode == FIND_PUPIL:
                self.angles = list(range(128))
            elif mode == FIND_IRIS:
                self.angles = list([*range(0, 16), *range(48, 80), *range(112, 128)])
        else:
            self.angles = angles
            
        if filter_model == None:
            # if mode == FIND_PUPIL:
            #     self.filter_model = CircleRANSAC(2, 62, 100)
            # elif mode == FIND_IRIS:
            #     self.filter_model = CircleRANSAC(2, 36, 100)
            self.filter_model = CircleRANSAC(2, 36, 100)
            
        else:
            self.filter_model = filter_model
            
        self.fitting_method = fitting_method

    def find(self, source):
        if self.mode==FIND_IRIS:
            mode_func = np.max
        elif self.mode ==FIND_PUPIL:
            mode_func = np.min
        else:
            raise KeyError("Wrong mode value. Use one of: irisloc.FIND_PUPIL, irisloc.FIND_IRIS")
        
        img = image_preprocessing(source)
        
        self.center_estimation = np.round(np.unravel_index(np.argmin(img), shape=img.shape))[::-1].astype(np.uint8)
        
        # POLAR WARP
        maxRad = 90.50966799187809
        img = img.astype(np.float32)
        polar = cv2.linearPolar(img, self.center_estimation, maxRad, cv2.WARP_FILL_OUTLIERS).astype(np.uint8)
        
        polar = cv2.medianBlur(polar, 9)
        
        # PEAKS SEARCHING
        # peaks = []
        if  self.peaks_find_method == POLAR_GRADIENT:
            # for ang in self.angles:
            #     deriv = np.gradient(polar[ang])
            #     deriv = np.clip(deriv, 0, None)
            #     maximas, _ = find_peaks(deriv)
            #     if len(maximas) > 2:
            #         ind_max = np.argpartition(deriv[maximas], -2)[-2:]
            #         peaks.append(mode_func(maximas[ind_max]))
            #     else:
            #         peaks.append(-1)
            peaks = grad_peaks(polar, self.angles, mode_func)
                    
        elif self.peaks_find_method == POLAR_CANNY:
            # for ang in self.angles:
            #     peaks.append(mode_func(np.argwhere(polar[ang] > 0)[:2]))
            peaks = canny_peaks(polar, self.angles, mode_func)
            
        elif self.peaks_find_method == POLAR_CURVEFIT:
            peaks = curvefit_peaks(polar, self.angles, mode_func)
                
        normalized_peaks = np.array([peak*maxRad/128 for peak in peaks])
        angles_rad = np.array(self.angles)*2*np.pi/128
        
        xs = np.full((len(self.angles),), self.center_estimation[1])+np.sin(angles_rad)*normalized_peaks
        ys = np.full((len(self.angles),), self.center_estimation[0])+np.cos(angles_rad)*normalized_peaks
        edge_points = np.vstack([ys, xs]).T
        
        # SIMPLE FILTERING
        edge_points = np.where(edge_points < 128, edge_points, -1)
        edge_points = edge_points[edge_points[:, 0] > 0]
        edge_points = edge_points[edge_points[:, 1] > 0]
        edge_points = edge_points[edge_points[:, 0] < 127]
        edge_points = edge_points[edge_points[:, 1] < 127]
        
        if self.fitting_method == fitCircleLS:
            if len(edge_points) < 3:
                return ((-1, -1), 0), []
        elif self.fitting_method in [fitEllipse, fitEllipseAMS, fitEllipseDirect]:
            if len(edge_points) < 5:
                return ((-1, -1), (0, 0), 0), []
        
        filtered_points = self.filter_model.filter(edge_points)
        
        if len(filtered_points) == 0:
            print()
            print("WARNING! Choosen filter_method in TwoGradMax returned 0 points.")
            print()
            if self.fitting_method == fitCircleLS:
                return (-1, -1), 0
            elif self.fitting_method in [fitEllipse, fitEllipseAMS, fitEllipseDirect]:
                return (-1, -1), (0, 0), 0

        return self.fitting_method(filtered_points.astype(np.float32))
    
    
# TODO CLASSICAL METHODS
class HoughTransform():
    def __init__(self, mode: int,
                 hough_param2=30,
                 preprocess: bool = True) -> None:
        assert mode in [FIND_PUPIL, FIND_IRIS], "Wrong mode value. Use one of: irisloc.FIND_PUPIL, irisloc.FIND_IRIS"
        self.mode = mode
        
        self.hough_param2=hough_param2
        
        self.preprocess = preprocess
        
        if mode == FIND_PUPIL:
            self.min_radius=8
            self.max_radius=36
        elif mode == FIND_IRIS:
            self.min_radius=36
            self.max_radius=64
        
    def find(self, source):
        if self.preprocess:
            img = image_preprocessing(source)
        
        # canny = cv2.Canny(img.astype(np.uint8), 128, 255)
        # circles = cv2.HoughCircles(canny.astype(np.uint8), cv2.HOUGH_GRADIENT, 
        #                            1, 1, 
        #                            param1=128, param2=self.hough_param2, 
        #                            minRadius=self.hough_minRadius, maxRadius=self.hough_maxRadius)
        
        # # print(circles)
        
        # if circles == None:
        #     return (-1, -1), 0, []
        
        # circles_sorted = np.array(sorted(circles[0], key=lambda x: x[2], reverse=True))
        
        # radii = circles_sorted[:, 2]
        # hist, bin_edges = np.histogram(radii, bins=2)
        # threshold_radius = bin_edges[1]
        
        # if self.mode == FIND_PUPIL:
        #     result = circles_sorted[np.where(circles_sorted[:,2] < threshold_radius)][0]
        # elif self.mode == FIND_IRIS:
        #     result = circles_sorted[np.where(circles_sorted[:,2] > threshold_radius)][0]
        # result = circles_sorted[len(circles_sorted)//2]
        # return (result[0], result[1]), result[2], circles_sorted

        # Edge detection
        edges = cv2.Canny(img.astype(np.uint8), 128, 255)

        height, width = edges.shape
        accumulator = np.zeros((height, width, self.max_radius - self.min_radius))

        # Hough Transform for circles
        for y in range(height):
            for x in range(width):
                if edges[y, x] > 0:  # Edge point
                    for r in range(self.min_radius, self.max_radius):
                        for theta in range(0, 360):
                            a = int(x - r * np.cos(theta * np.pi / 180))
                            b = int(y - r * np.sin(theta * np.pi / 180))
                            if 0 <= a < width and 0 <= b < height:
                                accumulator[b, a, r - self.min_radius] += 1

        # Find the best circle
        max_accumulator = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        best_y, best_x, best_r = max_accumulator
        best_r += self.min_radius

        return (best_x, best_y), best_r
    
     
class Daugman():
    def __init__(self, mode: int, preprocess: bool = True) -> None:
        assert mode in [FIND_PUPIL, FIND_IRIS], "Wrong mode value. Use one of: irisloc.FIND_PUPIL, irisloc.FIND_IRIS"
        self.mode = mode
        
        self.preprocess = preprocess
    
    def create_circle_mask(self, radius):
        mask = []
        for theta in np.linspace(0, 2 * np.pi, 128):
            a = int(radius * np.cos(theta))
            b = int(radius * np.sin(theta))
            mask.append((a, b, theta))
        return mask
    
    def compute_best_circle(self, grad_x, grad_y, height, width, radii_range):
        max_gradient_sum = 0
        best_circle = None

        for radius in radii_range:
            circle_mask = self.create_circle_mask(radius)
            for y in range(0, 127):
                for x in range(0, 127):
                    integral_sum = 0
                    count = 0

                    for point in circle_mask:
                        a = x + point[0]
                        b = y + point[1]

                        if 0 <= a < width and 0 <= b < height:
                            theta = point[2]
                            gradient = grad_x[b, a] * np.cos(theta) + grad_y[b, a] * np.sin(theta)
                            integral_sum += gradient
                            count += 1

                    if count > 0:
                        integral_sum /= count

                    if integral_sum > max_gradient_sum:
                        max_gradient_sum = integral_sum
                        best_circle = (x, y, radius)

        return best_circle

    def find(self, source):
        if self.preprocess:
            img = image_preprocessing(source)
        
        # Compute gradients using Sobel operator
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        if self.mode == FIND_PUPIL:
            radii_range = range(8, 36, 1)
        elif self.mode == FIND_IRIS:
            radii_range = range(42, 64, 1)

        # Apply the Integro-differential operator using the optimized function
        height, width = img.shape
        best_circle = self.compute_best_circle(grad_x, grad_y, height, width, radii_range)

        return (best_circle[0], best_circle[1]), best_circle[2]
     
     
# TODO CHOOSE ARCHITECHTURE AND LEARN OR FINETUNE an NN
# class NNLocalizer():
#     def __init__(self) -> None:
#          pass