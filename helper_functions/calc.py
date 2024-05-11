import math

import numpy as np
from numpy.polynomial.polynomial import Polynomial as P
from scipy import integrate, interpolate

from helper_functions.draw import *
from helper_functions.helper_func import get_image_name


# Attention: x_values and y_values were switched for calculating the centerline
# min_x: T1
#   |
#   |
#   |
# max_x: L5

def calculate_scd(x_values, y_values):
    """Calculates the SCD of a spine.
    SCD = sum of squared residuals / number of points

    Args:
        x_values: A list of x values.
        y_values: A list of y values.

    Returns:
        scd: The SCD of the spine.
        reg_line: The regression line of the spine.
    """
    reg_line, info = P.fit(x_values, y_values, 1, full=True)  # info = residuals, rank, singular_values, rcond
    sum_squared_residuals = info[0]
    scd = math.sqrt(sum_squared_residuals / len(x_values))
    return scd, reg_line

def calculate_sri(centerline, reg_line, low, up):
    """Calculates the integral of the difference between the centerline and the regression line for mt angle
    Args:
        centerline: The centerline of the spine.
        reg_line: The regression line of the spine.
    """
    integral = integrate.quad(lambda x: (centerline(x) - reg_line(x))**2, low, up)[0]
    sri = math.sqrt(integral / (up - low))
    return sri

def calculate_angle(min_normal, max_normal, flag):
    """Calculates the angle between two normals.
    A.B = |A| |B| cosθ,  |A| = |B| = 1
    A.B = cosθ

    Args:
        min_normal: The minimum normal (normalized).
        max_normal: The maximum normal (normalized).

    Returns:
        deg_angle: The angle between the two normals in degrees
        flag: 0 if no error occurred, 3 if an error occurred
    """
    try:
        rad_angle = np.arccos(np.dot(min_normal, max_normal))
        if np.isnan(rad_angle):
            flag = 3
            deg_angle = np.nan
        else:
            deg_angle = np.rad2deg(rad_angle)
    except:
        print("NaN value")
        flag = 3
        deg_angle = np.nan

    return deg_angle, flag

def calculate_angle_list(normals):
    """Calculates the angles between a list of normals.
    Args:
        normals: A list of normals.

    Returns:
        angles: A list of angles.
        flag: 0 if no error occurred, 3 if an error occurred
    """
    flag = 0
    angles = []
    for i in range(len(normals)-1):
        if np.isnan(normals[i][0]) or np.isnan(normals[i+1][0]):
            angles.append(np.nan)
            continue
        angle, flag = calculate_angle(normals[i], normals[i+1], flag)
        if flag != 0:
            angles.append(np.nan)
        else:
            angles.append(angle)
    
    return angles, flag


def calculate_normal(spl, x, slope, show=False):
    """Calculates the normal of a function.

    Args:
        spl: The polynomial function.
        x: The x value.
        slope: The slope of the function at x.
        show: If true, the function will be drawn. default = False

    Returns:
        normal: The normal vector of the function at x (normalized)
    """
    if x == -1: # if no point was found
        return np.array([np.nan, np.nan])
    normal = np.array([-slope, 1])

    if show:
        interval = np.linspace(x - 100, x + 100, 200)
        plt.plot(interval, spl(interval), label='f(x)')
        plt.plot(x, spl(x), 'o')
        numbers = np.linspace(-50, 50, 100)
        plt.plot(x + numbers * normal[0], spl(x) + numbers * normal[1], label='normal')
        plt.plot(x + numbers * 1, spl(x) + numbers * slope, label='tangent')

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')  # set equal axis otherwise 90° angle is not visible
        plt.legend()
        plt.title('Normal')
        plt.show()
    return normal / np.linalg.norm(normal)


def calculate_normal_list(spl, points, show=False):
    """Calculates the normals of a list of points."""
    normals = []
    for point in points:
        normal = calculate_normal(spl, point[0], point[1], show)
        normals.append(normal.tolist())
        
    return normals


def interpolate_spline(x_values, y_values, params, show=False):
    """Interpolates a spline function from a list of x and y values."""
    typ = params['spline_type']
    value = params['spline_val']
    typ_smooth = params['spline_type_smooth']
    value_smooth = params['spline_val_smooth']
    smoothing_fac = params['smooth']
    cand = params['spl_cand']
    endpiece_index = params['endpiece_index']

    max_x = max(x_values)
    min_x = min(x_values)
        
    s = len(x_values)*smoothing_fac
    smoothed_spl = interpolate.UnivariateSpline(x_values, y_values, s=s)
    smoothed_x = np.linspace(min_x, max_x, int((max_x - min_x)*cand))
    smoothed_y = smoothed_spl(smoothed_x)
    
    both_spl = calculate_spline(typ_smooth, value_smooth, smoothed_x, smoothed_y)
    single_spl = calculate_spline(typ, value, x_values, y_values)
    spl = decide(both_spl, single_spl, (x_values[endpiece_index],x_values[-1]))
    
    # draw function
    if show:
        x2 = np.linspace(int(min_x), int(max_x), int(max_x - min_x))
        y2 = spl(x2)
        plt.plot(x_values, y_values, 'o', x2, y2)
        plt.title('Spline fit')
        plt.show()

    return spl

def calculate_spline(typ, value, x, y):
    """Calculates a spline function from a list of x and y values."""
    if typ == 'cubic':
        if value == "both2":
            cond = ((2, 0.0), (2, 0.0))
        if value == "left2":
            cond = ((2, 0.0), (1, 0.0))
        elif value == "right2":
            cond = ((1, 0.0), (2, 0.0))
        else:
            cond = ((1, 0.0), (1, 0.0))
        spl = interpolate.CubicSpline(x, y, bc_type=cond)
    elif typ == 'b':
        spl = interpolate.BSpline(x, y, value)
    elif typ == 'akima':
        spl = interpolate.Akima1DInterpolator(x, y)
    elif typ == 'pchip':
        spl = interpolate.PchipInterpolator(x, y)
    return spl

def decide(both_spl, single_spl, endpiece):
    """Decides which spline function to use, based on the maximum derivative."""
    array = np.linspace(endpiece[0], endpiece[1], 100)
    both_deriv = both_spl.derivative(1)
    single_deriv = single_spl.derivative(1)
    both_max = max([abs(both_deriv(x)) for x in array])
    single_max = max([abs(single_deriv(x)) for x in array])
    if both_max > single_max:
        return single_spl
    else:
        return both_spl   


def interpolate_polynomial(x_values, y_values, degree=6, show=False):
    """Interpolates a polynomial function from a list of x and y values."""
    poly, info = P.fit(x_values, y_values, degree, full=True)
    max_x = max(x_values)
    min_x = min(x_values)

    # draw function
    if show:
        x2 = np.linspace(int(min_x), int(max_x), int(max_x - min_x))
        y2 = poly(x2)
        plt.plot(x_values, y_values, 'o', x2, y2)
        plt.title('Polynomial fit')
        plt.show()

    return poly

def calculate_max_min_slope(spl, min_x, max_x, centroids_list, params):
    points = []
    div_dist = params['dist']
    if type(div_dist) == str:
        div_dist = len(centroids_list)
    dist = (max_x -min_x) / div_dist
    pt_min = min_x + params['pt_dist']*dist
    tl_max = max_x - params['tl_dist']*dist
    mt_min = pt_min + params['dist_min']*dist
    mt_max = tl_max - params['dist_max']*dist
    candidates = np.linspace(min_x, max_x, params['num_cand'])

    points.append(find_max_inclination(spl, candidates, mt_min, mt_max, positive=True))  # point1 increasing /
    points.append(find_max_inclination(spl, candidates, mt_min, mt_max, positive=False)) # point2 decreasing \
    if points[0][0] > points[1][0]: # x_min --- point3 --- point2 --- point1 --- point4 --- x_max => / \ / \
        pt_max = points[1][0] 
        tl_min = points[0][0] 
        points.insert(0, find_max_inclination(spl, candidates, pt_min, pt_max, positive=True, pt=True))  # point3 increasing /
        points.append(find_max_inclination(spl, candidates, tl_min, tl_max, positive=False, tl= True))  # point4 decreasing \
    else:  # x_min --- point3 --- point1 --- point2 --- point4 --- x_max => \ / \ /
        pt_max = points[0][0] 
        tl_min = points[1][0]
        points.insert(0, find_max_inclination(spl, candidates, pt_min, pt_max, positive=False, pt= True)) # point3 decreasing \
        points.append(find_max_inclination(spl, candidates, tl_min, tl_max, positive=True, tl=True)) # point4 increasing /

    points.sort(key=lambda x: x[0])
    interval_start = points[1][0]
    interval_end = points[2][0]

    return points, interval_start, interval_end


def find_max_inclination(spl, candidates, min_x, max_x, positive, pt=False, tl=False):
    fst_der = spl.derivative(1)
    array = [x for x in candidates if min_x <= x <= max_x]
    point = []
    if positive:
        max_list = [(x, fst_der(x).item()) for x in array if fst_der(x) >= 0]
        if max_list:
            point.append(max(max_list, key=lambda x: x[1]))
    else:
        min_list = [(x, fst_der(x).item()) for x in array if fst_der(x) <= 0]
        if min_list:
            point.append(min(min_list, key=lambda x: x[1]))    
    if pt:
        point.append((min_x, fst_der(min_x).item()))
    elif tl:
        point.append((max_x, fst_der(max_x).item()))
    else:
        middle = (max_x -min_x)/2 + min_x
        point.append((middle, fst_der(middle).item())) #  only chosen if nothing is found
    
    return point[0]


def calculate_values(image_path, image, centroids_list, params, use_images):
    """Calculates the values for each image."""
    centroids_list.sort(key=lambda x: x[1]) # sort after y-values
    y_values = [x[0] for x in centroids_list]
    x_values = [x[1] for x in centroids_list]
    min_x = min(x_values)
    max_x = max(x_values)
    
    if not x_values or not y_values:
        return None, image, 1
    
    for i in range(len(x_values) - 1):
        if x_values[i] > x_values[i + 1]:
            return None, image, 1

    scd, reg_line = calculate_scd(x_values, y_values)
    spl = interpolate_spline(x_values, y_values, params)
    
    points, interval_start, interval_end = calculate_max_min_slope(spl, min_x, max_x, centroids_list, params)
    normals = calculate_normal_list(spl, points)
    angles, flag = calculate_angle_list(normals)
    
    sri = calculate_sri(spl, reg_line, interval_start, interval_end)

    image_name = get_image_name(image_path)
    values_dict = {'image_path': image_name, 'info': None, 'angle_pt': angles[0], 'angle_mt': angles[1], 'angle_tl': angles[2],
                   'sri': sri, 'scd': scd,
                   'centroids': centroids_list, 'normals': normals, 'points': points, 'min_x': min_x,
                   'max_x': max_x, 'spl': spl}

    # draw result
    if use_images:
        x2 = np.linspace(math.ceil(min_x), math.floor(max_x), int(max_x - min_x))
        y2 = spl(x2)
        res_image_angle = draw_centerline(image, y2, x2)
        res_image_angle = draw_normals(res_image_angle, normals, points, spl)
    else:
        res_image_angle = None
    
    return values_dict, res_image_angle, flag

