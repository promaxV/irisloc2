import cv2
import numpy as np


def delete_constant_tail(lst, const = None):
    if const == None:
        const = lst[-1]
    for i, x in enumerate(lst[::-1]):
        if x != const:
            break
    return lst[:-1-i]

def eye_func_half(x_arr,
                mid_const_radius, mid_const_y, 
                right_parabola_a, 
                right_const_lpx, right_const_length, right_const_y, 
                right_line_x, right_line_y, right_line_a):

        return_list = []
        mid_const_center_x = 0
        for x in x_arr:
            if x <= mid_const_center_x + mid_const_radius:
                return_list.append(mid_const_y)
            elif x <= right_const_lpx:
                right_parabola_b = (right_const_y - mid_const_y - right_parabola_a*(right_const_lpx - mid_const_center_x - mid_const_radius)**2)/((right_const_lpx-mid_const_center_x-mid_const_radius))
                return_list.append(right_parabola_a*(x - mid_const_center_x - mid_const_radius)**2 + right_parabola_b*(x - mid_const_center_x - mid_const_radius) + mid_const_y)
            elif x <= right_const_lpx + right_const_length:
                return_list.append(right_const_y)
            elif x <= right_line_x:
                right_half_parabola_a = (right_const_y-right_line_y)/((right_line_x-right_const_lpx-right_const_length)**2)
                right_half_parabola_b = 2*right_half_parabola_a*(right_const_lpx+right_const_length-right_line_x)
                return_list.append(right_half_parabola_a*(x - right_const_lpx - right_const_length)**2 + right_half_parabola_b*(x - right_const_lpx - right_const_length) + right_const_y)
            else: return_list.append(right_line_a*x + right_line_y - right_line_a*right_line_x)
            
        return return_list
    
def autoScaleABC(image):
    min_v = image.min()
    max_v = image.max()
    a = 255/(max_v-min_v)
    b = 255*(1-max_v/(max_v-min_v))
    return np.clip(a*image+b, 0, 255)
    
def sharpen(image, weight=0.5, gauss_ksize=0, gauss_sigmaX = 5, gamma=0):
    blured_eye = cv2.GaussianBlur(image, (gauss_ksize, gauss_ksize),  gauss_sigmaX)
    return cv2.addWeighted(image, 1+weight, blured_eye, -weight, gamma)