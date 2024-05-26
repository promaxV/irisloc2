import numpy as np

from abc import ABC, abstractmethod

from fitting import fitCircleLS
from cv2 import fitEllipse, fitEllipseAMS, fitEllipseDirect

from fitting import ellipse_distance

SMED_FILTER_CENTER_MEAN = 0

class FilterModel(ABC):
    pass
    
class SimpleMedian(FilterModel):
    def __init__(self, sigma_k, center = SMED_FILTER_CENTER_MEAN) -> None:
        self.sigma_k = sigma_k
        self.center = center

    def filter(self, points):
        # if self.center == SMED_FILTER_CENTER_MEAN:
        self.center = points.mean(axis=0)
        distances = np.linalg.norm(points - self.center, axis=1)
        d_median = np.median(distances)
        d_std = distances.std()
        return points[np.abs(distances - d_median) < self.sigma_k*d_std]

class CircleRANSAC(FilterModel):
    def __init__(self, distance, inliers, max_iter, __return_best_candidate = False) -> None:
        self.distance = distance
        self.inliers = inliers
        self.max_iter = max_iter

        self.__return_best_candidate = __return_best_candidate
        
    def filter(self, points):
        best_mse = 10000
        best_sample_ind = []
        
        random_samples = []
        init_list = list(range(len(points)))
        iterations = (self.max_iter//(len(points)//3*3)+1)*3
        random_seeds = list(range(iterations))
        for random_seed in random_seeds:
            np.random.seed(random_seed)
            np.random.shuffle(init_list)
            random_samples.append(np.array(init_list[:len(init_list)//3*3]).reshape(-1, 3))
        
        random_samples = np.array(random_samples).reshape(-1, 3)
        
        # for _ in range(self.max_iter):
        for random_sample_indeces in random_samples:
            # random_sample_indeces = np.random.choice(len(points), 3, replace=False)
            
            circle_center, circle_radius = fitCircleLS(points[random_sample_indeces])
            inliers_ind = []
            distances = []
            for i in range(len(points)):
                point_distance = abs(np.linalg.norm(points[i] - circle_center) - circle_radius)
                if point_distance < self.distance:
                    inliers_ind.append(i)
                    distances.append(point_distance)
            if len(inliers_ind) >= self.inliers:
                
                mse = np.power(distances, 2).mean()
                if mse < best_mse:
                    best_mse = mse
                    best_center, best_radius, best_sample_ind = circle_center, circle_radius, inliers_ind
                    
        if not self.__return_best_candidate:
            return points[best_sample_ind]
        else:
            circle_center, circle_radius = fitCircleLS(points[best_sample_ind])
            return points[best_sample_ind], (best_center, best_radius, best_mse)

class EllipseRANSAC(FilterModel):
    def __init__(self, distance, inliers, max_iter, __return_best_candidate = False) -> None:
        self.distance = distance
        self.inliers = inliers
        self.max_iter = max_iter
        
        self.__return_best_candidate = __return_best_candidate
        
    def filter(self, points):
        best_mse = 10000
        best_sample_ind = []
        for _ in range(self.max_iter):
            random_sample_indeces = np.random.choice(len(points), 5, replace=False)
            ((centx,centy), (width,height), angle) = fitEllipse(points[random_sample_indeces].astype(np.float32))
            inliers_ind = []
            distances = []
            for i in range(len(points)):
                point_distance = ellipse_distance(points[i], centx, centy, width, height, np.deg2rad(angle))
                if point_distance < self.distance:
                    inliers_ind.append(i)
                    distances.append(point_distance)
            if len(inliers_ind) >= self.inliers:
                mse = np.power(distances, 2).mean()
                if mse < best_mse:
                    best_mse = mse
                    best_center, best_width, best_height, best_angle, best_sample_ind = (centx, centy), width, height, angle, inliers_ind

        if not self.__return_best_candidate:
            return np.array(points[best_sample_ind]).astype(np.float32)
        else:
            ((centx,centy), (width,height), angle) = fitEllipse(points[random_sample_indeces].astype(np.float32))
            return np.array(points[best_sample_ind]).astype(np.float32), (best_center, best_width, best_height, best_angle)