import cv2
import numpy as np
import imutils
import math
from sklearn.cluster import KMeans
import shapely


def to_blur(img, kernel=(5, 5)):
    out_image = img.copy()
    return cv2.GaussianBlur(out_image, kernel, 0)


def to_normal(img, debug=False):
    out_image = img.copy()
    if len(img.shape) == 3:
        out_image = cv2.cvtColor(out_image, cv2.COLOR_BGR2GRAY)

    if debug:
        cv2.imshow("Normal", out_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (
        (out_image - np.min(out_image))
        / np.max(np.max(out_image) - np.min(out_image))
        * 255.0
    ).astype(np.uint8)


def to_quant(img, k=8, normalize=False, debug=False):
    image = cv2.imread(img) if isinstance(img, str) else img
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Z = img_rgb.reshape((-1, 3)).astype(np.float32)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct quantized image
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(img_rgb.shape)

    if normalize:
        quantized = to_normal(quantized)

    if debug:
        cv2.imshow("Original", img_rgb)
        cv2.imshow("Quantized", quantized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return quantized


def get_longest_contour(contour):
    l = 0
    idx = -1
    for _i, _n in enumerate(contour):
        if _n.shape[0] > l:
            l = _n.shape[0]
            idx = _i
    return idx, l


def get_border_color(
    img_1chan, min_thresh=200, max_thresh=255, min_num_consecutive=8, reverse_walk=False
):
    # walk diag checking pixels:

    l = min(*img_1chan.shape[:2])

    num_consecutive = 0
    avg_border_val = []

    # start in center, walk out
    for _n in range(l // 2, 0, -1):
        idx = _n if not reverse_walk else l // 2 + (l // 2 - _n)
        if img_1chan[idx, idx] >= min_thresh and img_1chan[idx, idx] <= max_thresh:
            avg_border_val.append(img_1chan[idx, idx])
            num_consecutive += 1
        else:
            num_consecutive = 0
            avg_border_val.clear()

        if num_consecutive >= min_num_consecutive:
            v, c = np.unique(avg_border_val, return_counts=True)
            print("found border at ", idx, v, c)
            m_idx = np.argmax(c)
            # return mode
            return v[m_idx]
    print(
        f"Unable to find border consec count {num_consecutive} avg border vals {avg_border_val}"
    )
    return None


def remove_directional_shadow(image, kernel_size=31):
    """
    Reduces directional shadowing from an RGB image using illumination estimation.

    Parameters:
        image (ndarray): Input BGR image
        kernel_size (int): Size for Gaussian smoothing

    Returns:
        result (ndarray): Lighting-corrected image
    """
    # Convert to grayscale to estimate intensity gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Estimate illumination using heavy Gaussian blur
    illumination = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Avoid division by zero
    illumination = np.where(illumination == 0, 1, illumination).astype(np.float32)

    # Normalize each channel independently using illumination
    channels = cv2.split(image.astype(np.float32))
    corrected_channels = [
        np.clip((c / illumination) * 128, 0, 255).astype(np.uint8) for c in channels
    ]

    # Merge channels back to BGR
    result = cv2.merge(corrected_channels)
    return result


def get_border_mask(
    img,
    quant_k=10,
    border_color=None,
    border_pixels=20,
    threshold_width=1,
    blur=True,
    clahe=False,
    remove_shadows=False,
    reverse_border_walk=False,
    debug=False,
):
    image = cv2.imread(img) if isinstance(img, str) else img
    orig = image.copy()
    if remove_shadows:
        image = remove_directional_shadow(image)
    if clahe:
        image = apply_clahe(image)
    if blur:
        image = to_blur(image)

    c = None

    quant_k = quant_k if isinstance(quant_k, list) else [quant_k]

    while len(quant_k) and c is None:
        next_k = quant_k.pop()
        border_img = to_quant(image, next_k, debug=debug)
        border_img = to_normal(border_img, debug=debug)

        if border_color is not None:
            print ('setting color to provided border color',border_color)
            c = border_color
        else:
            c = get_border_color(
                border_img,
                min_thresh=230,
                min_num_consecutive=border_pixels,
                reverse_walk=reverse_border_walk,
            )

    if c is None:
        return None, None, None, None

    print('c is ',c)
    if c >= 255:
        c = 254
    print(
        f"border color {c}, corner {border_img[0,0]} lower thresh {c - threshold_width} upper thresh {c+ threshold_width}"
    )
    # _, mask = cv2.threshold(border_img, c - 1, 255, cv2.THRESH_BINARY)
    # TOZERO 0 if <= thresh
    _, mask1 = cv2.threshold(
        border_img, c - 1 - threshold_width, 255, cv2.THRESH_TOZERO
    )
    # TOZERO INV 0 if > thresh
    _, mask2 = cv2.threshold(
        border_img, c + threshold_width, 255, cv2.THRESH_TOZERO_INV
    )

    mask = cv2.bitwise_and(mask1, mask2)

    if debug:
        cv2.imshow("Original", orig)
        cv2.imshow("processed", border_img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return border_img, mask, orig, c


def apply_clahe(img):
    image = cv2.imread(img) if isinstance(img, str) else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def get_max_contour(contours):
    max_ctr_area = 0
    max_ctr_area_idx = 0
    for _i, contour in enumerate(contours):
        ctr_area = cv2.contourArea(contour)
        if ctr_area > max_ctr_area:
            max_ctr_area = ctr_area
            max_ctr_area_idx = _i

    print("idx", max_ctr_area_idx)
    return contours[max_ctr_area_idx], max_ctr_area_idx, max_ctr_area


def get_uniform_contour(contours, min_area=300):
    straightest_contour = None
    straightest_contour_idx = 0
    contour_area = 0
    lowest_deviation = float("inf")
    for _i, contour in enumerate(contours):
        if len(contour) >= 2:
            # Fit a line to the contour
            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Project contour points onto the line and compute deviation
            direction = np.array([vx, vy]).reshape(-1)
            origin = np.array([x, y]).reshape(-1)

            deviations = []
            for point in contour:
                pt = point[0]
                pt_vec = pt - origin
                proj_length = np.dot(pt_vec, direction)
                proj_point = origin + proj_length * direction
                deviation = np.linalg.norm(pt - proj_point)
                deviations.append(deviation)

            mean_deviation = np.mean(deviations)

            # Update if this is the straightest so far
            if (
                cv2.contourArea(contour) > min_area
                and mean_deviation < lowest_deviation
            ):
                lowest_deviation = mean_deviation
                straightest_contour = contour
                straightest_contour_idx = _i
                contour_area = cv2.contourArea(contour)

    print("winning cont area ", contour_area)
    return straightest_contour, straightest_contour_idx, contour_area


def extend_line(pt1, pt2, num):
    direction = pt2 - pt1
    direction = direction / np.linalg.norm(direction)  # Normalize

    # Calculate new points
    extended_pt1 = pt1 - direction * num
    extended_pt2 = pt2 + direction * num

    return tuple(extended_pt1.astype(np.int64)), tuple(extended_pt2.astype(np.int64))


def get_straight_lines(img, min_len=200, max_gap=10):
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )  # Adjust minLineLength as needed

    ret_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        ret_lines.append(((x1, y1), (x2, y2)))

    return ret_lines


# quantized_img = quantize_image_kmeans("path_to_your_image.jpg", k=6, debug=True)
def preprocess_border(img, min_thresh=220, max_thresh=255, debug=False):
    image = cv2.imread(img) if isinstance(img, str) else img
    orig = image.copy()
    # image = imutils.resize(image, width=1000)

    # Convert to LAB and apply CLAHE for contrast enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    # Threshold lightness to isolate white border
    _, mask = cv2.threshold(l_eq, min_thresh, max_thresh, cv2.THRESH_BINARY)

    # Morph cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if debug:
        cv2.imshow("Original", orig)
        cv2.imshow("mask", mask)
        cv2.imshow("clahe", l_eq)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, mask, contours, orig


def angle_from_vectors(l1, l2):
    v1_x = l1[1][0] - l1[0][0]
    v1_y = l1[1][1] - l1[0][1]
    v2_x = l2[1][0] - l2[0][0]
    v2_y = l2[1][1] - l2[0][1]

    """Calculates the angle between two vectors using the dot product."""
    dot_product = v1_x * v2_x + v1_y * v2_y
    magnitude1 = math.sqrt(v1_x**2 + v1_y**2)
    magnitude2 = math.sqrt(v2_x**2 + v2_y**2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0  # Handle zero-length vectors

    cosine_theta = dot_product / (magnitude1 * magnitude2)

    # Ensure cosine_theta is within valid range [-1, 1] for acos
    cosine_theta = max(-1.0, min(1.0, cosine_theta))

    angle_radians = math.acos(cosine_theta)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


def get_pt_intersections(lines, exts):

    intersection_pts = []
    for _i in range(len(lines)):  # tuple of tuple
        for _j in range(len(lines)):  # tuple of tuple
            if _j == _i:
                continue

            theta = angle_from_vectors(lines[_i], lines[_j])
            if abs(theta) < 80 or abs(theta) > 100:
                continue
            xdiff = (
                lines[_i][0][0] - lines[_i][1][0],
                lines[_j][0][0] - lines[_j][1][0],
            )
            ydiff = (
                lines[_i][0][1] - lines[_i][1][1],
                lines[_j][0][1] - lines[_j][1][1],
            )

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div == 0:
                continue  # Lines are parallel

            d = (
                det(
                    (lines[_i][0][0], lines[_i][0][1]),
                    (lines[_i][1][0], lines[_i][1][1]),
                ),
                det(
                    (lines[_j][0][0], lines[_j][0][1]),
                    (lines[_j][1][0], lines[_j][1][1]),
                ),
            )
            x = det(d, (xdiff[0], xdiff[1])) / div
            y = det(d, (ydiff[0], ydiff[1])) / div
            if x < exts[0][0] or x > exts[1][0] or y < exts[0][1] or y > exts[1][1]:
                continue
            intersection_pts.append((int(x), int(y)))

    # reduce to 4
    X = np.asarray(intersection_pts)
    kmeans = KMeans(
        n_clusters=4,
    ).fit(X)

    def sort_to_corners(pts):
        # Step 1: Sum and difference of coordinates
        sums = pts.sum(axis=1).astype(np.int32)  # x + y
        diffs = np.diff(pts, axis=1).astype(np.int32)  # x - y

        # Step 2: Assign corners
        top_left = pts[np.argmin(sums)]
        bottom_right = pts[np.argmax(sums)]
        top_right = pts[np.argmin(diffs)]
        bottom_left = pts[np.argmax(diffs)]
        return (
            top_left.astype(np.int32),
            top_right.astype(np.int32),
            bottom_left.astype(np.int32),
            bottom_right.astype(np.int32),
        )

    return sort_to_corners(kmeans.cluster_centers_)


def get_border_width(img_1ch, mask_val=255):
    top_border_width = 0
    bottom_border_width = 0
    image_width = 0

    inborder = False
    inimage = False
    img_half_width = img_1ch.shape[1] // 2

    for _n in range(img_1ch.shape[0]):
        if img_1ch[_n, img_half_width] == 255:
            inframe = True


def warp_pts(*args, m):
    # Apply warp
    return cv2.perspectiveTransform(
        np.asarray([*args]).reshape(-1, 1, 2).astype(np.float32), m
    ).astype(np.int32)


def warp_image(img, tl, tr, bl, br, extend=0):
    # Source points (e.g. detected corners on the card in original image)
    src_pts = np.array(
        [tl, tr, bl, br],  # Top-left  # Top-right  # Bottom-left  # Bottom-right
        dtype=np.float32,
    )

    src_pts[0] += [-extend, -extend]
    src_pts[1] += [extend, -extend]
    src_pts[2] += [-extend, extend]
    src_pts[3] += [extend, extend]

    # Destination points: define where you want the corners mapped to
    width = int(min(br[0] - bl[0], tr[0] - tl[0]))
    height = int(min(br[1] - tr[1], bl[1] - tl[1]))

    dst_pts = np.array(
        [
            [0, 0],  # Top-left
            [width - 1, 0],  # Top-right
            [0, height - 1],  # Bottom-left
            [width - 1, height - 1],  # Bottom-right
        ],
        dtype=np.float32,
    )

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply warp
    exts = warp_pts(tl, tr, bl, br, m=matrix)
    return cv2.warpPerspective(img, matrix, (width, height)), np.squeeze(exts), matrix


def get_edges_corners(img, extend=10):

    tl_c = img[0 : extend * 2, 0 : extend * 2]
    tr_c = img[0 : extend * 2, -extend * 2 :]
    bl_c = img[-extend * 2 :, 0 : extend * 2]
    br_c = img[-extend * 2 :, -extend * 2 :]
    t_b = img[0 : extend * 2, :]
    b_b = img[-extend * 2 :, :]
    l_b = img[:, : extend * 2]
    r_b = img[:, -extend * 2 :]
    return (tl_c, tr_c, bl_c, br_c), (t_b, b_b, l_b, r_b)


def get_major_lines(img, num_lines, line_axis):

    axis = 0 if line_axis == 1 else 1

    sums = img.sum(axis=axis)
    ind = np.argpartition(sums, -num_lines)[-num_lines:]

    if line_axis == 0:
        return [[(0, _n), (img.shape[1], _n)] for _n in ind]
    return [[(_n, 0), (_n, img.shape[0])] for _n in ind]


def get_min_max_lines(lines, max, axis):
    ret_idx = -1
    cur_val = -1 if max else 1e10
    for _i, _n in enumerate(lines):
        if max and _n[0][axis] > cur_val:
            cur_val = _n[0][axis]
            ret_idx = _i
        elif not max and _n[0][axis] < cur_val:
            cur_val = _n[0][axis]
            ret_idx = _i
    return lines[ret_idx]


def get_ideal_edge_contour(warp_exts, pixels=10):
    # warp exts TL TR BL BR

    # top
    t = np.asarray(
        [
            [
                warp_exts[0],
                warp_exts[1],
                [warp_exts[1][0], warp_exts[1][1] + pixels],
                [warp_exts[0][0], warp_exts[1][1] + pixels],
            ]
        ]
    )
    b = np.asarray(
        [
            [
                [warp_exts[2][0], warp_exts[2][1] - pixels],
                [warp_exts[3][0], warp_exts[3][1] - pixels],
                warp_exts[3],
                warp_exts[2],
            ]
        ]
    )
    l = np.asarray(
        [
            [
                warp_exts[0],
                [warp_exts[0][0] + pixels, warp_exts[0][1]],
                [warp_exts[2][0] + pixels, warp_exts[2][1]],
                warp_exts[2],
            ]
        ]
    )
    r = np.asarray(
        [
            [
                [warp_exts[1][0] - pixels, warp_exts[1][1]],
                warp_exts[1],
                warp_exts[3],
                [warp_exts[3][0] - pixels, warp_exts[3][1]],
            ]
        ]
    )

    return t, b, l, r


def get_ideal_corner_contour(img, max_x, max_y, pixels=10):
    x_line = get_min_max_lines(get_major_lines(img, 3, line_axis=0), max_x, 1)
    y_line = get_min_max_lines(get_major_lines(img, 3, line_axis=1), max_y, 0)
    corner_pt = (y_line[0][0], x_line[0][1])

    if max_x and max_y:
        other_pt = (corner_pt[0] - pixels, corner_pt[1] - pixels)
    elif max_x and not max_y:
        other_pt = (corner_pt[0] + pixels, corner_pt[1] - pixels)
    elif not max_x and max_y:
        other_pt = (corner_pt[0] - pixels, corner_pt[1] + pixels)
    else:
        other_pt = (corner_pt[0] + pixels, corner_pt[1] + pixels)

    return (
        np.asarray(
            [
                corner_pt,
                (corner_pt[0], other_pt[1]),
                other_pt,
                (other_pt[0], corner_pt[1]),
            ]
        )
        .reshape(4, 1, 2)
        .astype(np.int32)
    )


def get_poly_from_pts(pts):
    pts = np.squeeze(pts)
    return [shapely.Point(*_n) for _n in pts]


def score_corner(img_cntr, ideal_cntr):
    img_poly = shapely.Polygon(get_poly_from_pts(img_cntr))
    ideal_poly = shapely.Polygon(get_poly_from_pts(ideal_cntr))

    area_diff = shapely.difference(ideal_poly, img_poly).area

    return (ideal_poly.area - area_diff) / ideal_poly.area


def score_edges(img_cntr, ideal_cntr):
    img_poly = shapely.Polygon(get_poly_from_pts(img_cntr))
    ideal_poly = shapely.Polygon(get_poly_from_pts(ideal_cntr))

    area_diff = shapely.difference(ideal_poly, img_poly).area

    return (ideal_poly.area - area_diff) / ideal_poly.area


def score_card(image_path, debug=False):
    results = {}
    image, img_mask, orig, contours, c_idx, opt_k, opt_c = opt_border(
        image_path, max_iter=10, clahe=True, debug=debug
    )

    c_img = np.full(image.shape, 0.0)
    best_c_img = cv2.drawContours(c_img, contours, c_idx, (255, 0, 0), 2)
    border_lines = get_straight_lines(
        best_c_img.astype(np.uint8), min_len=110, max_gap=20
    )
    border_ext_lines = []
    for _n in border_lines:
        p1, p2 = _n
        p1, p2 = extend_line(np.asarray(p1), np.asarray(p2), 500)
        border_ext_lines.append((p1, p2))
    int_pts = get_pt_intersections(
        border_ext_lines, exts=[[0, 0], [image.shape[1], image.shape[0]]]
    )
    tl, tr, bl, br = int_pts
    warp_img, warp_exts, warp_m = warp_image(orig, tl, tr, bl, br, extend=18)

    # TL TR BL BR
    # Edges: T B L R
    clip_ext = 50
    corners, edges = get_edges_corners(warp_img, extend=clip_ext)
    border_corner_color = None

    corner_clips = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]

    for _i, _n_corner in enumerate(corners):
        _, mask, _, color = get_border_mask(
            _n_corner,
            quant_k=list(range(15, 3, -2)),
            border_color=border_corner_color,
            border_pixels=20,
            blur=False,
            debug=False,
            clahe=True,
            remove_shadows=False,
            threshold_width=2,
        )
        if color is None:  # try again but other direction
            print("trying reverse border walk")
            _, mask, _, color = get_border_mask(
                _n_corner,
                quant_k=list(range(15, 3, -2)),
                border_color=border_corner_color,
                blur=False,
                debug=debug,
                clahe=True,
                remove_shadows=False,
                threshold_width=2,
                reverse_border_walk=True,
            )
        if color is None:
            _, mask, _, color = get_border_mask(
                _n_corner,
                quant_k=list(range(15, 3, -2)),
                border_color=opt_c,
                blur=False,
                debug=debug,
                clahe=True,
                remove_shadows=False,
                threshold_width=2,
            )
        if border_corner_color is None:
            border_corner_color = color  # TODO this is hacky, fix
        contours = get_contours(mask)
        _ctr, _idx, _area = get_max_contour(contours)
        contour_img = np.full(mask.shape, 0).astype(np.uint8)
        cv2.drawContours(contour_img, contours, _idx, (255, 0, 0), 1)
        ideal_contour = get_ideal_corner_contour(
            contour_img, *corner_clips[_i], pixels=15
        )
        corner_score = score_corner(contours[_idx], ideal_contour)

        results.setdefault("corner-scores", {}).setdefault(_i, corner_score)
        results.setdefault("corner-contour", {}).setdefault(_i, contours[_idx])
        results.setdefault("corner-ideal-contour", {}).setdefault(_i, ideal_contour)
        results.setdefault("corner-shape", {}).setdefault(_i, mask.shape)

    # return T B L R
    edge_ideal_contour = list(get_ideal_edge_contour(warp_exts, pixels=30))
    edge_ideal_straight_contour = list(get_ideal_edge_contour(warp_exts, pixels=10))
    # shift points to local coords (T L are okay)
    print("wapr shape", img_mask.shape)
    print("ideals", edge_ideal_contour[1])
    edge_ideal_contour[1] -= np.asarray([0, warp_img.shape[0] - (clip_ext * 2)])
    edge_ideal_contour[3] -= np.asarray([warp_img.shape[1] - (clip_ext * 2), 0])
    edge_ideal_straight_contour[1] -= np.asarray(
        [0, warp_img.shape[0] - (clip_ext * 2)]
    )
    edge_ideal_straight_contour[3] -= np.asarray(
        [warp_img.shape[1] - (clip_ext * 2), 0]
    )
    print("ideals", edge_ideal_contour[1])

    for _i, _n_edge in enumerate(edges):
        _, mask, _, color = get_border_mask(
            _n_edge,
            quant_k=list(range(15, 3, -2)),
            border_color=border_corner_color,
            blur=False,
            debug=debug,
            clahe=True,
            remove_shadows=False,
            threshold_width=2,
        )
        if color is None:  # try again but other direction
            print("trying reverse border walk")
            _, mask, _, color = get_border_mask(
                _n_edge,
                quant_k=list(range(15, 3, -2)),
                border_color=border_corner_color,
                blur=False,
                debug=debug,
                clahe=True,
                remove_shadows=False,
                threshold_width=2,
                reverse_border_walk=True,
            )

        if border_corner_color is None:
            border_corner_color = color  # TODO this is hacky, fix
        contours = get_contours(mask)
        _ctr, _idx, _area = get_max_contour(contours)

        edge_score = score_corner(contours[_idx], edge_ideal_contour[_i])
        edge_straight_score = score_corner(
            contours[_idx], edge_ideal_straight_contour[_i]
        )
        results.setdefault("edge-shape", {}).setdefault(_i, edges[_i].shape)
        results.setdefault("edge-contour", {}).setdefault(_i, contours[_idx])
        results.setdefault("edge-scores", {}).setdefault(_i, edge_score)
        results.setdefault("edge-straight-scores", {}).setdefault(
            _i, edge_straight_score
        )
        results.setdefault("edge-ideal-contour", {}).setdefault(
            _i, edge_ideal_contour[_i]
        )
        results.setdefault("edge-straight-ideal-contour", {}).setdefault(
            _i, edge_ideal_straight_contour[_i]
        )

    print("corner results TL TR BL BR", results["corner-scores"])
    print("straight edge results TBLR", results["edge-straight-scores"])
    print("edge results TBLR", results["edge-scores"])
    return orig, image, img_mask, warp_img, results


def opt_border(
    image_path, min_c_area=1e5, max_c_area=2.0e6, max_iter=5, clahe=False, debug=False
):
    _k = 15
    _border_pixels = 20
    _border_walk_reverse = False

    for _n in range(max_iter):
        print("trying k ", _k)
        image, mask, orig, border_c = get_border_mask(
            image_path,
            border_pixels=_border_pixels,
            quant_k=_k,
            clahe=clahe,
            reverse_border_walk=_border_walk_reverse,
            debug=debug,
        )
        if image is None:
            if _n % 2 == 0:
                _border_walk_reverse = not _border_walk_reverse
                print(
                    f"did not find border, reversing border walk : {_border_walk_reverse}"
                )

            else:
                _border_pixels -= 3
                print(f"did not find border, reducing border pixels:{_border_pixels}")
            continue
        contours = get_contours(mask)
        c, c_idx, c_area = get_uniform_contour(contours, min_area=5e3)
        print("area check ", c_area, min_c_area, max_c_area)

        if c_area >= min_c_area and c_area <= max_c_area:
            return image, mask, orig, contours, c_idx, _k, border_c
        elif c_area > max_c_area:
            _k += 2
        else:
            _k -= 2
            _k = _k if _k > 0 else 1
    return None, None, None, None, None, None, None


def find_nameplate(img):
    image = cv2.imread(img) if isinstance(img, str) else img
    orig = image.copy()


def find_best_right_angle(contour, window=5):
    """
    Finds the point in the contour with the angle closest to 90 degrees.

    Parameters:
        contour (ndarray): OpenCV contour (Nx1x2 array).
        window (int): Number of points to skip on either side for angle calculation.

    Returns:
        tuple: (best_deviation, best_point_index, angle_at_best)
    """
    contour = contour[:, 0, :]
    n_points = len(contour)

    best_deviation = float("inf")
    best_index = None
    angle_at_best = None

    for i in range(window, n_points - window):
        pt_prev = contour[i - window]
        pt_curr = contour[i]
        pt_next = contour[i + window]

        vec1 = pt_prev - pt_curr
        vec2 = pt_next - pt_curr

        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_product))

        deviation = abs(90 - angle_deg)

        if deviation < best_deviation:
            best_deviation = deviation
            best_index = i
            angle_at_best = angle_deg

    return best_deviation, best_index, angle_at_best


def get_contours(img):
    contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(contours)


def preprocess_nameplate(img, mask_thresh=220, mask_max=255, mask_width=1, debug=False):
    image = cv2.imread(img) if isinstance(img, str) else img
    orig = image.copy()

    if image.ndim == 3:
        image = np.mean(image, axis=-1).astype(np.uint8)

    _, mask = cv2.threshold(
        image, mask_thresh + mask_width, mask_max, cv2.THRESH_BINARY
    )
    # _, mask = cv2.threshold(mask, mask_thresh - mask_width, mask_max, cv2.THRESH_BINARY_INV)

    # Morph cleanup
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if debug:
        cv2.imshow("Original", orig)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, mask, contours, orig


def preprocess_border2(img, mask_thresh=220, mask_max=255, mask_width=1, debug=False):
    image = cv2.imread(img) if isinstance(img, str) else img
    orig = image.copy()

    if image.ndim == 3:
        image = np.mean(image, axis=-1).astype(np.uint8)
    # image = imutils.resize(image, width=1000)

    # Convert to LAB and apply CLAHE for contrast enhancement
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # print ('s',np.histogram(s))
    # print ('v',np.histogram(v))
    #
    # Threshold lightness to isolate white border
    # > thresh become 0

    # TOZERO_INV 0 if > thresh
    _, mask = cv2.threshold(
        image, mask_thresh + mask_width, mask_max, cv2.THRESH_TOZERO_INV
    )
    # TOZERO  0 if <= thresh
    _, mask = cv2.threshold(mask, mask_thresh - mask_width, mask_max, cv2.THRESH_TOZERO)

    # Morph cleanup
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if debug:
        cv2.imshow("Original", orig)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image, mask, contours, orig


def scan_border(
    gray_image, num_axis_0, num_axis_1, mask_min_thresh=210, mask_max_thresh=255
):
    vert_areas = []
    horz_areas = []
    h_contours = []
    v_contours = []
    h_delta = gray_image.shape[0] // num_axis_0
    v_delta = gray_image.shape[1] // num_axis_1
    for _n_0 in np.linspace(0, gray_image.shape[0], num_axis_0):
        img = np.zeros(gray_image.shape).astype(gray_image.dtype)
        img[int(_n_0) : int(_n_0 + h_delta), :] = gray_image[
            int(_n_0) : int(_n_0 + h_delta), :
        ]
        l_contours = imutils.grab_contours(
            cv2.findContours(
                img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )
        for _n_c in l_contours:
            a = cv2.contourArea(_n_c)
            horz_areas.append(a)
            h_contours.append(_n_c)
    for _n_1 in np.linspace(0, gray_image.shape[1], num_axis_1):
        img = np.zeros(gray_image.shape).astype(gray_image.dtype)
        img[:, int(_n_1) : int(_n_1 + v_delta)] = gray_image[
            :, int(_n_1) : int(_n_1 + v_delta)
        ]
        l_contours = imutils.grab_contours(
            cv2.findContours(
                img,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
        )
        for _n_c in l_contours:
            a = cv2.contourArea(_n_c)
            vert_areas.append(a)
            v_contours.append(_n_c)
    h_idx0 = np.argpartition(np.asarray(horz_areas[: len(horz_areas) // 2]), -1)[-1]
    h_idx1 = (
        np.argpartition(np.asarray(horz_areas[len(horz_areas) // 2 :]), -1)[-1]
        + len(horz_areas) // 2
    )
    v_idx0 = np.argpartition(np.asarray(vert_areas[: len(vert_areas) // 2]), -1)[-1]
    v_idx1 = (
        np.argpartition(np.asarray(vert_areas[len(vert_areas) // 2 :]), -1)[-1]
        + len(vert_areas) // 2
    )
    print("maxs", h_idx0, h_idx1, v_idx0, v_idx1)
    print(
        "areas",
        np.asarray(horz_areas)[[h_idx0, h_idx1]],
        np.asarray(vert_areas)[[v_idx0, v_idx1]],
    )

    return (
        h_contours[h_idx0],
        h_contours[h_idx1],
        v_contours[v_idx0],
        v_contours[v_idx1],
    )
