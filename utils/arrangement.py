import numpy as np
from shapely.geometry import Polygon


def is_left(node_pt, line_pts):
    """
    Check which side does a node lie to a line
    :param node_pt: coordinates of the node to be checked
    :param line_pts: coordinates of source -> target terminal nodes
    :return: int, positive means to the left side, negative the right side, and zero means on the line
    """
    a_x, a_y = node_pt
    (b_x, b_y), (c_x, c_y) = line_pts
    return np.sign((a_x - b_x) * (b_y - c_y) + (a_y - b_y) * (c_x - b_x))


def representative_point(face):
    """
    Return the representative point of a given skgeom.Face
    :param face: skgeom.arrangement.Face
    :return: (float, float), coordinates of the representative points
    """
    # Convert the skgeom.arrangement.Face into a shapely.geometry.Polygon
    face_poly_pos = []
    half_edge_circulator = face.outer_ccb
    first_half_edge = None
    next_half_edge = next(half_edge_circulator)
    while next_half_edge != first_half_edge:
        if first_half_edge is None:
            first_half_edge = next_half_edge
            face_poly_pos.append((first_half_edge.source().point().x(), first_half_edge.source().point().y()))
        else:
            face_poly_pos.append((next_half_edge.source().point().x(), next_half_edge.source().point().y()))
        next_half_edge = next(half_edge_circulator)
    face_poly = Polygon(face_poly_pos)
    # Get the representative point
    rep_point = face_poly.representative_point()
    return float(rep_point.x), float(rep_point.y)
