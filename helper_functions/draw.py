import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_points(frame, points_list, color=(0, 0, 255)):
    """ Draws a list of points on a frame."""
    new_frame = frame.copy()
    for centroid in points_list:
        int_centroid = (int(centroid[0]), int(centroid[1]))
        cv2.circle(new_frame, int_centroid, 5, color, 3)
    return new_frame


def draw_bboxes(frame, bbox_list, color=(0, 255, 0)):
    """ Draws a list of bounding boxes on a frame.
    Args:
        frame: The frame to draw the bounding boxes on.
        bbox_list: A list of bounding boxes. [x1, y2, x2, y2]
        color: The color of the bounding boxes. default = green
    """
    new_frame = frame.copy()
    for bbox in bbox_list:
        start_point = (int(bbox[0]), int(bbox[1]))  # upper-left corner
        end_point = (int(bbox[2]), int(bbox[3]))  # bottom-right corner
        cv2.rectangle(new_frame, start_point, end_point, color, 2)
    return new_frame


def draw_centerline(frame, x_values, y_values, color=(255, 0, 0)):
    """draws a centerline on a frame
    Args:
        frame: The frame to draw the centerline on.
        x_values: A list of x values for the centerline.
        y_values: A list of y values for the centerline.
        color: The color of the centerline. default = red

    Attention: x_values and y_values were switched for calculating the centerline. Switch it back on method call.
    """
    new_frame = frame.copy()
    points = np.asarray([x_values, y_values]).T.astype(int)
    cv2.polylines(new_frame, [points], False, color, 3)
    return new_frame


def draw_normal(image, normal, x, y, color=(0, 255, 0)):
    """draws a normal on a frame

    Args:
        image: The frame to draw the normal on.
        normal: The normal-vector to draw.
        x: The x value of a point on the normal.
        y: The y value of a point on the normal.
        color: The color of the normal. default = green

    Returns:
        new_frame: The frame with the normal drawn on it.

    Attention: x_values and y_values were switched for calculating the centerline. Switch it back on method call.
                Normal stayed the same. Adjusted in method.
    """
    if normal[0] == 'NaN':
        return image

    start_point = (int(x + normal[1] * (-200)), int(y + normal[0] * (-200)))
    end_point = (int(x + normal[1] * 200), int(y + normal[0] * 200))

    return cv2.line(image, start_point, end_point, color, 3)


def draw_normals(image, normals, points, spl, color=(0, 255, 0)):
    new_frame = image.copy()
    for i in range(len(normals)):
        new_frame = draw_normal(new_frame, normals[i], spl(points[i][0]), points[i][0], color)
    return new_frame


def draw_angle_to_value(sri, angle, value_str, title, num):
    """Draws a scatter plot of angle to value."""
    plt.subplot(1, 2, num)
    plt.scatter(angle, sri)
    plt.xlabel('sum of angles in °', fontsize=15)
    plt.ylabel(value_str, fontsize=15)
    plt.title(title, fontsize=17)



