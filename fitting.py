import math
import numpy as np
from scipy.optimize import leastsq

def fitCircleLS(points):
    x = points[:, 0]
    y = points[:, 1]
    
    def calcR(xc, yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2(center):
        Ri = calcR(*center)
        return Ri - Ri.mean()

    center_estimation = (x.mean(), y.mean())
    center, _ = leastsq(f_2, center_estimation)

    return center, calcR(*center).mean()

def ellipse_distance(x, centx, centy, width, height, angle, tolerance = 1e-8, max_iterations = 1000, __return_full_info = False):
        """
        Finds the minimum distance between the specified point and the ellipse
        using Newton's method.
        """
        
        def angle_to_rotation_matrix(angle):
            """
            Returns the rotation matrix for the ellipse's rotation.
            """
            a = math.cos(angle)
            b = math.sin(angle)
            return np.array([[a, -b], [b, a]])
        
        x = np.asarray(x)
        r = angle_to_rotation_matrix(angle)
        x2 = np.dot(r.T, x - [centx, centy])
        t = math.atan2(x2[1], x2[0])
        a = 0.5 * width
        b = 0.5 * height
        
        # If point is inside ellipse, generate better initial angle based on vertices
        if (x2[0] / a)**2 + (x2[1] / b)**2 < 1:
            ts = np.linspace(0, 2 * math.pi, 24, endpoint = False)
            xe = a * np.cos(ts)
            ye = b * np.sin(ts)
            delta = x2 - np.column_stack([xe, ye])
            t = ts[np.argmin(np.linalg.norm(delta, axis = 1))]
            
        iterations = 0
        error = tolerance
        errors = []
        ts = []
                
        while error >= tolerance and iterations < max_iterations:
            cost = math.cos(t)
            sint = math.sin(t)
            x1 = np.array([a * cost, b * sint])
            xp = np.array([-a * sint, b * cost])
            xpp = np.array([-a * cost, -b * sint])
            delta = x1 - x2
            dp = np.dot(xp, delta)
            dpp = np.dot(xpp, delta) + np.dot(xp, xp)
            t -= dp / dpp
            error = abs(dp / dpp)
            errors.append(error)
            ts.append(t)
            iterations += 1
        
        ts = np.array(ts)
        errors = np.array(errors)
        y = np.linalg.norm(x1 - x2)
        success = error < tolerance and iterations < max_iterations
        if __return_full_info:
            return dict(x = t, y = y, error = error, iterations = iterations, success = success, xs = ts,  errors = errors)
        else:
            return y