# This file will contain the card detection logic.
import cv2
import numpy as np

class CardDetector:
    def __init__(self, gaussian_blur_kernel_size=(9, 9), canny_threshold1=30, canny_threshold2=100): # Best so far
        self.gaussian_blur_kernel_size = gaussian_blur_kernel_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        # Expected aspect ratio: width/height or height/width - Increased tolerance to 10% (reverted)
        self.expected_aspect_ratio_range = ( (2.5 / 3.5) * 0.90, (2.5 / 3.5) * 1.10 )
        self.expected_aspect_ratio_range_inv = ( (3.5 / 2.5) * 0.90, (3.5 / 2.5) * 1.10 )
        self.min_contour_area_factor = 0.05 # Used by contour method

        # Parameters for Hough Line Transform
        self.hough_rho = 1  # distance resolution in pixels of the Hough grid
        self.hough_theta = np.pi / 180  # angular resolution in radians of the Hough grid
        self.hough_threshold = 60  # minimum number of votes (intersections in Hough grid cell)
        self.hough_min_line_length = 150  # minimum number of pixels making up a line - Increased
        self.hough_max_line_gap = 25  # maximum gap in pixels between connectable line segments - Decreased
        self.hough_min_dominant_line_length_ratio = 0.15 # Min length for a line to be considered dominant (ratio of img dim)


    def _load_and_preprocess_image(self, image_path):
        """Loads an image, converts to grayscale, and applies Gaussian blur."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, self.gaussian_blur_kernel_size, 0)
        return image, blurred_image

    def find_card_edges(self, image_path):
        original_image, preprocessed_image = self._load_and_preprocess_image(image_path)

        # Perform Canny edge detection
        edges_map = cv2.Canny(preprocessed_image, self.canny_threshold1, self.canny_threshold2)

        # Morphological Closing operation
        kernel_morph = np.ones((5,5),np.uint8) # Back to 5x5 kernel. Removed OPEN.
        closed_edges_map = cv2.morphologyEx(edges_map, cv2.MORPH_CLOSE, kernel_morph)

        # Find contours on the closed edge map
        contours, _ = cv2.findContours(closed_edges_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by area (descending) and filter
        if not contours:
            return original_image, None # No contours found

        # Sort contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = original_image.shape[0] * original_image.shape[1]
        min_area = image_area * self.min_contour_area_factor

        candidate_contours = []
        for contour in contours:
            # Check minimum area
            if cv2.contourArea(contour) < min_area:
                break # Since contours are sorted, no need to check smaller ones

            # Approximate the contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Standard epsilon

            # The card should be a quadrilateral
            if len(approx) == 4:
                # Check aspect ratio
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                is_valid_aspect_ratio = (self.expected_aspect_ratio_range[0] <= aspect_ratio <= self.expected_aspect_ratio_range[1]) or \
                                        (self.expected_aspect_ratio_range_inv[0] <= aspect_ratio <= self.expected_aspect_ratio_range_inv[1])

                if is_valid_aspect_ratio:
                    # print(f"Found candidate contour with area: {cv2.contourArea(approx)}, aspect_ratio: {aspect_ratio:.3f}")
                    candidate_contours.append(approx)

        # print(f"Total candidate contours found: {len(candidate_contours)}")
        selected_contour = None
        if not candidate_contours:
            # print("No candidate contours passed initial filters.")
            return original_image, None

        # Filter out contours that are too large (e.g., >95% of image area) if there are multiple candidates
        # and then pick the largest remaining.
        # If only one candidate, use it even if it's very large.
        image_total_area = original_image.shape[0] * original_image.shape[1]
        max_allowed_area_ratio = 0.95

        if len(candidate_contours) > 1:
            valid_candidates_after_size_filter = []
            for c in candidate_contours:
                if cv2.contourArea(c) / image_total_area < max_allowed_area_ratio:
                    valid_candidates_after_size_filter.append(c)

            if valid_candidates_after_size_filter: # If any contours are left after filtering
                # print(f"Found {len(valid_candidates_after_size_filter)} candidates after size filter.")
                selected_contour = max(valid_candidates_after_size_filter, key=cv2.contourArea)
                # print(f"Selected contour area after size filter: {cv2.contourArea(selected_contour)}")
            else: # All candidates were too large, pick the smallest of the "too large" ones
                  # This means it picks the one closest to the max_allowed_area_ratio from above.
                # print("All candidates were too large. Picking the smallest of the large ones.")
                selected_contour = min(candidate_contours, key=cv2.contourArea)
        else: # Only one candidate
            # print("Only one candidate contour found. Selecting it.")
            selected_contour = candidate_contours[0]

        # if selected_contour is not None:
            # print(f"Final selected contour area: {cv2.contourArea(selected_contour)}")

        if selected_contour is not None:
            # Reshape contour to be a list of points and order them
            points = selected_contour.reshape(4, 2)
            ordered_points = self._order_points(points)
            return original_image, ordered_points
        else:
            return original_image, None


    def _find_card_edges_hough(self, original_image, preprocessed_image):
        """
        Finds card edges using Canny edge detection followed by Hough Line Transform.
        This is an alternative to the contour-based method.
        """
        # Perform Canny edge detection (same as before)
        edges_map = cv2.Canny(preprocessed_image, self.canny_threshold1, self.canny_threshold2)

        # Perform Hough Line Transform (Probabilistic)
        lines = cv2.HoughLinesP(edges_map, self.hough_rho, self.hough_theta, self.hough_threshold,
                                np.array([]), self.hough_min_line_length, self.hough_max_line_gap)

        if lines is None:
            print("Hough Transform found no lines.")
            return None # No lines found

        print(f"Hough Transform found {len(lines)} lines.")

        # Actual logic will be much more complex:
        # 1. Filter and categorize lines (horizontal/vertical)
        # 2. Find intersections
        # 3. Select 4 intersections that form a valid card quadrilateral
        # 4. Order points

        # For initial testing, let's try to draw detected lines on a copy of the original image
        # if you want to visualize this during testing phase of this function.
        # line_image = original_image.copy()
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.imwrite("output/hough_lines_detected.png", line_image)


        # This part needs to be implemented properly.
        # For now, if lines are found, we can't return proper corners yet.
        # So, for the purpose of integrating this into the main flow,
        # we'll return None until the full logic is in place.
        # This will make the test fail, which is expected at this stage.

        # ---- START OF COMPLEX LOGIC TO BE IMPLEMENTED ----

        # Filter lines: by angle to get horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        angle_tolerance_degrees = 5 # Stricter angle tolerance

        for line_segment in lines:
            x1, y1, x2, y2 = line_segment[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi

            # Normalize angle to be between 0 and 180
            if angle < 0:
                angle += 180

            if (abs(angle) < angle_tolerance_degrees or abs(angle - 180) < angle_tolerance_degrees or abs(angle-0) < angle_tolerance_degrees ): # Horizontal
                 horizontal_lines.append(line_segment[0])
            elif (abs(angle - 90) < angle_tolerance_degrees or abs(angle + 90) < angle_tolerance_degrees): # Vertical
                 vertical_lines.append(line_segment[0])

        # print(f"Found {len(horizontal_lines)} raw H lines, {len(vertical_lines)} raw V lines.")

        # Merge lines
        # Using default tolerances for now, these might need tuning.
        merged_horizontal_lines = self._merge_horizontal_lines(horizontal_lines, y_tolerance=10, x_min_overlap_pixels=20)
        merged_vertical_lines = self._merge_vertical_lines(vertical_lines, x_tolerance=10, y_min_overlap_pixels=20)

        print(f"Found {len(merged_horizontal_lines)} merged H lines, {len(merged_vertical_lines)} merged V lines.")

        # The condition for proceeding should be based on merged_lines now.
        # Need at least 2 H and 2 V lines to define a quadrilateral by unique top/bottom and left/right.
        if not merged_horizontal_lines or not merged_vertical_lines or \
           len(merged_horizontal_lines) < 1 or len(merged_vertical_lines) < 1: # Need at least 1 of each to pick extremes
            print("Not enough distinct merged horizontal OR vertical lines.")
            return None

        # New strategy: Select 4 dominant lines (top, bottom, left, right) from MERGED lines
        # and find their intersections.

        # Filter for reasonably long lines to be candidates for dominant lines
        # This length check should now apply to MERGED lines.
        # min_dominant_line_length_ratio is now self.hough_min_dominant_line_length_ratio
        img_h, img_w = preprocessed_image.shape[:2]

        # No separate "long_horizontal_lines" filtering here, _get_candidate_edge_lines handles length.

        num_edge_candidates = 5 # Number of candidates to consider for each edge (as per plan step 1)

        # Pass the full list of merged lines to _get_candidate_edge_lines.
        # It will handle sorting by position and length based on is_primary_edge.
        image_shape_for_candidates = preprocessed_image.shape[:2] # (height, width)

        candidate_top_lines = self._get_candidate_edge_lines(
            merged_horizontal_lines, True, image_shape_for_candidates, num_edge_candidates,
            self.hough_min_dominant_line_length_ratio, is_primary_edge=True)

        candidate_bottom_lines = self._get_candidate_edge_lines(
            merged_horizontal_lines, True, image_shape_for_candidates, num_edge_candidates,
            self.hough_min_dominant_line_length_ratio, is_primary_edge=False)

        candidate_left_lines = self._get_candidate_edge_lines(
            merged_vertical_lines, False, image_shape_for_candidates, num_edge_candidates,
            self.hough_min_dominant_line_length_ratio, is_primary_edge=True)

        candidate_right_lines = self._get_candidate_edge_lines(
            merged_vertical_lines, False, image_shape_for_candidates, num_edge_candidates,
            self.hough_min_dominant_line_length_ratio, is_primary_edge=False)

        print(f"Candidate lines: Top({len(candidate_top_lines)}), Bottom({len(candidate_bottom_lines)}), Left({len(candidate_left_lines)}), Right({len(candidate_right_lines)})")
        print(f"  Top candidates: {candidate_top_lines}")
        print(f"  Bottom candidates: {candidate_bottom_lines}")
        print(f"  Left candidates: {candidate_left_lines}")
        print(f"  Right candidates: {candidate_right_lines}")


        if not (candidate_top_lines and candidate_bottom_lines and candidate_left_lines and candidate_right_lines):
            print("Not enough candidate lines for one or more edges.")
            return None

        best_quad = None
        max_score = -1 # Or some other initial low score

        for top_l in candidate_top_lines:
            for bottom_l in candidate_bottom_lines:
                # Ensure top is above bottom
                if ((top_l[1] + top_l[3]) / 2) >= ((bottom_l[1] + bottom_l[3]) / 2):
                    continue
                for left_l in candidate_left_lines:
                    for right_l in candidate_right_lines:
                        # Ensure left is to the left of right
                        if ((left_l[0] + left_l[2]) / 2) >= ((right_l[0] + right_l[2]) / 2):
                            continue

                        tl = self._get_line_intersection(top_l, left_l)
                        tr = self._get_line_intersection(top_l, right_l)
                        bl = self._get_line_intersection(bottom_l, left_l)
                        br = self._get_line_intersection(bottom_l, right_l)

                        if not (tl and tr and bl and br):
                            continue

                        current_corners = np.array([tl, tr, br, bl], dtype="float32")

                        # Basic geometric checks (convexity is harder here, rely on ordering and aspect ratio)
                        # Aspect ratio
                        x_coords = current_corners[:, 0]
                        y_coords = current_corners[:, 1]
                        w = np.max(x_coords) - np.min(x_coords)
                        h = np.max(y_coords) - np.min(y_coords)

                        if w < 10 or h < 10: continue # Too small / degenerate

                        aspect_ratio = float(w) / h
                        is_valid_aspect = (self.expected_aspect_ratio_range[0] <= aspect_ratio <= self.expected_aspect_ratio_range[1]) or \
                                          (self.expected_aspect_ratio_range_inv[0] <= aspect_ratio <= self.expected_aspect_ratio_range_inv[1])
                        if not is_valid_aspect:
                            continue

                        # Area check
                        area = w * h # cv2.contourArea(current_corners.reshape(-1,1,2).astype(np.int32)) doesn't work directly for non-contours
                        if not (area > self.min_contour_area_factor * img_h * img_w and area < 0.90 * img_h * img_w) : # Max area 90%
                            continue

                        # SCORING - Placeholder: for now, just take the first valid one.
                        # TODO: Implement proper scoring (e.g., edge support) - This is it!
                        # Score the current valid quadrilateral
                        # Note: raw_hough_lines is 'lines' in this scope
                        # The _order_points is important before scoring if side order matters for _calculate_edge_support_score
                        # However, _score_candidate_quad takes TL,TR,BR,BL ordered points.
                        # The current_corners are already TL,TR,BR,BL from intersection.
                        # Let's pass the ordered points to the scorer.

                        ordered_current_corners = self._order_points(current_corners.copy())

                        chosen_lines_for_quad = {'top': top_l, 'bottom': bottom_l, 'left': left_l, 'right': right_l}
                        current_score = self._score_candidate_quad(ordered_current_corners, lines,
                                                                  preprocessed_image.shape, chosen_lines_for_quad)

                        if current_score > max_score:
                            max_score = current_score
                            best_quad = ordered_current_corners
                            # print(f"New best quad: score {max_score:.2f}, corners {best_quad.tolist()}") # Redundant with print in _score_candidate_quad
                        # End of scoring block

        if best_quad is not None:
            print(f"--- Selected best quad with final score {max_score:.2f} ---")
            return best_quad

        print("No suitable quadrilateral found after checking combinations and scoring.")
        return None
        # ---- END OF NEW QUADRILATERAL GENERATION & SELECTION ----

    def _calculate_edge_support_score(self, quad_side, raw_hough_lines, angle_similarity_thresh_rad=np.deg2rad(10), dist_thresh=15): # Original tolerances
        """Calculates how well a quadrilateral side is supported by raw Hough lines."""
        p1, p2 = quad_side
        # Calculate properties of the quad side
        side_dx = p2[0] - p1[0]
        side_dy = p2[1] - p1[1]
        side_length = np.sqrt(side_dx**2 + side_dy**2)
        if side_length == 0: return 0
        side_angle = np.arctan2(side_dy, side_dx)

        support_score = 0
        for raw_line_seg in raw_hough_lines:
            x1, y1, x2, y2 = raw_line_seg[0]

            # Raw line properties
            raw_dx = x2 - x1
            raw_dy = y2 - y1
            raw_length = np.sqrt(raw_dx**2 + raw_dy**2)
            if raw_length == 0: continue
            raw_angle = np.arctan2(raw_dy, raw_dx)

            # Check angle similarity (parallelism)
            angle_diff = abs(raw_angle - side_angle)
            angle_diff = min(angle_diff, np.pi - angle_diff) # Handle 180-degree equivalence
            if angle_diff > angle_similarity_thresh_rad:
                continue

            # Check proximity of raw line segment to the (infinite) line of quad_side
            # Distance from point (x0,y0) to line Ax+By+C=0 is |Ax0+By0+C|/sqrt(A^2+B^2)
            # Line for quad_side: side_dy * x - side_dx * y + (side_dx * p1[1] - side_dy * p1[0]) = 0
            # A = side_dy, B = -side_dx, C = side_dx * p1[1] - side_dy * p1[0]
            # Check midpoint of raw_line_seg
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            numerator = abs(side_dy * mid_x - side_dx * mid_y + (side_dx * p1[1] - side_dy * p1[0]))
            denominator = side_length # which is sqrt(A^2+B^2) = sqrt(side_dy^2 + (-side_dx)^2)
            if denominator == 0: continue # Should not happen if side_length > 0

            distance = numerator / denominator
            if distance > dist_thresh:
                continue

            # Optional: Check for overlap along the direction of the side
            # This is more complex; for now, proximity and parallelism are the main checks.
            # A simple check: does the midpoint of the raw line, when projected onto the quad side's line,
            # fall within the segment p1-p2?
            # Vector p1p2 = (side_dx, side_dy)
            # Vector p1_to_mid = (mid_x - p1[0], mid_y - p1[1])
            # Dot product: (p1_to_mid . p1p2) / side_length^2
            # If this value 'k' is between 0 and 1, midpoint projection is on segment.
            dot_product = (mid_x - p1[0]) * side_dx + (mid_y - p1[1]) * side_dy
            k = dot_product / (side_length**2)

            if 0 <= k <= 1: # Projection of midpoint is on the segment
                 support_score += 1 # Count supporting lines

        return support_score # Return count, not density


    def _score_candidate_quad(self, quad_corners, raw_hough_lines, image_shape, chosen_lines):
        """
        Scores a candidate quadrilateral based on geometry and edge support.
        chosen_lines: dict {'top': top_l, 'bottom': bottom_l, 'left': left_l, 'right': right_l}
        """
        # 1. Geometric checks (already partially done by caller, but can be re-verified)
        # For now, assume caller did basic aspect/area. Convexity also important.
        # A simple check for convexity: points should be ordered TL, TR, BR, BL.
        # Cross product of (P2-P1)x(P0-P1) and (P0-P3)x(P2-P3) etc. should have same sign.
        # Or, use cv2.isContourConvex after reshaping. For now, we skip detailed convexity check here.

        score = 0

        # 2. Edge Support Score
        # quad_corners are [tl, tr, br, bl]
        sides = [
            (quad_corners[0], quad_corners[1]), # Top
            (quad_corners[1], quad_corners[2]), # Right
            (quad_corners[2], quad_corners[3]), # Bottom
            (quad_corners[3], quad_corners[0])  # Left
        ]

        total_edge_support = 0
        for side in sides:
            total_edge_support += self._calculate_edge_support_score(side, raw_hough_lines)

        # For now, score is just total edge support.
        # Could be weighted by area, aspect ratio deviation, etc.
        score = total_edge_support

        # Normalize by perimeter to avoid favoring huge quads?
        # perimeter = sum(np.sqrt( (s[1][0]-s[0][0])**2 + (s[1][1]-s[0][1])**2 ) for s in sides)
        # if perimeter > 0:
        #     score /= perimeter

        # --- New Geometric Scores ---
        tl, tr, br, bl = quad_corners[0], quad_corners[1], quad_corners[2], quad_corners[3]

        # Side lengths
        len_top = np.linalg.norm(tr - tl)
        len_bottom = np.linalg.norm(br - bl)
        len_left = np.linalg.norm(bl - tl)
        len_right = np.linalg.norm(br - tr)

        if any(l == 0 for l in [len_top, len_bottom, len_left, len_right]): # Avoid division by zero
            return 0 # Invalid quad for geometric scoring

        # Angles (in radians)
        angle_top = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
        angle_bottom = np.arctan2(bl[1] - br[1], bl[0] - br[0]) # Reversed for parallelism (BL-BR)
        angle_left = np.arctan2(bl[1] - tl[1], bl[0] - tl[0])
        angle_right = np.arctan2(br[1] - tr[1], br[0] - tr[0])

        # 1. Parallelism Score (0 to 1, 1 is perfectly parallel)
        # Angle difference, normalized to be between 0 and PI/2
        # Smaller diff is better. Score = 1 - (diff / (PI/2))
        # Max diff we tolerate for "parallel" could be e.g. 10 degrees (0.17 rad)
        max_angle_dev_parallel = np.deg2rad(10)

        diff_top_bottom = abs(angle_top - angle_bottom)
        diff_top_bottom = min(diff_top_bottom, np.pi - diff_top_bottom) # handles angles 180 deg apart
        score_parallel_h = max(0, 1 - diff_top_bottom / max_angle_dev_parallel)

        diff_left_right = abs(angle_left - angle_right)
        diff_left_right = min(diff_left_right, np.pi - diff_left_right)
        score_parallel_v = max(0, 1 - diff_left_right / max_angle_dev_parallel)

        parallelism_score = (score_parallel_h + score_parallel_v) / 2

        # 2. Perpendicularity Score (0 to 1, 1 is perfectly perpendicular)
        # Target diff is PI/2. Score = 1 - (abs(actual_diff - PI/2) / (PI/2))
        max_angle_dev_perp = np.deg2rad(10) # Max deviation from 90 deg

        # Top-Left
        diff_tl = abs(abs(angle_top - angle_left) - np.pi/2)
        diff_tl = min(diff_tl, np.pi - diff_tl) # if angles were > PI apart
        score_perp_tl = max(0, 1 - diff_tl / max_angle_dev_perp)

        # Top-Right (angle_right vs angle_top)
        diff_tr = abs(abs(angle_right - angle_top) - np.pi/2)
        diff_tr = min(diff_tr, np.pi - diff_tr)
        score_perp_tr = max(0, 1 - diff_tr / max_angle_dev_perp)

        # (Can add all 4 corners, or just average two)
        perpendicularity_score = (score_perp_tl + score_perp_tr) / 2 # Simplified for now

        # 3. Length Similarity Score (0 to 1, 1 is equal length)
        # Ratio of shorter to longer. Score = min(l1,l2)/max(l1,l2)
        score_len_h = min(len_top, len_bottom) / max(len_top, len_bottom)
        score_len_v = min(len_left, len_right) / max(len_left, len_right)
        length_similarity_score = (score_len_h + score_len_v) / 2

        # Weights for combining scores (these need tuning)
        w_edge_support = 0.1 # Reduced
        w_parallelism = 0.3  # Increased
        w_perpendicularity = 0.3 # Increased
        w_length_similarity = 0.3 # Increased
        w_coverage = 0.2 # New weight for coverage

        # Coverage Score (placeholder - needs the original merged lines that formed this quad)
        # This requires passing top_l, bottom_l, left_l, right_l to the scoring function
        # For now, let's set coverage_score to 0 or 1 and adjust _find_card_edges_hough to pass lines.
        # This change is too large for one diff if I modify the call signature now.
        # I will calculate a pseudo coverage here based on quad_corners for now,
        # assuming the quad_corners directly reflect the chosen merged lines' intersection.
        # This is not ideal, as quad_corners are just points.
        # The actual merged lines should be used.

        # Coverage Score Calculation
        top_l = chosen_lines['top']
        bottom_l = chosen_lines['bottom']
        left_l = chosen_lines['left']
        right_l = chosen_lines['right']

        quad_sides_details = [
            (quad_corners[0], quad_corners[1], top_l, True),  # Top side TL-TR
            (quad_corners[1], quad_corners[2], right_l, False), # Right side TR-BR
            (quad_corners[3], quad_corners[2], bottom_l, True),# Bottom side BL-BR (reversed to match BR-BL for consistency if needed, but length is abs)
            (quad_corners[0], quad_corners[3], left_l, False)  # Left side TL-BL
        ]

        total_coverage_metric = 0
        num_valid_sides_for_coverage = 0

        for p1_qc, p2_qc, merged_line_segment, is_h in quad_sides_details:
            qc_side_len = np.linalg.norm(np.array(p2_qc) - np.array(p1_qc))
            if qc_side_len < 1e-6: continue # Avoid division by zero for zero-length sides

            num_valid_sides_for_coverage +=1

            # Calculate overlap between the merged line segment and the actual quad side
            if is_h: # Horizontal side, compare X-ranges. Merged line: [x1,y,x2,y]
                overlap = max(0, min(merged_line_segment[2], max(p1_qc[0], p2_qc[0])) - \
                                   max(merged_line_segment[0], min(p1_qc[0], p2_qc[0])))
            else: # Vertical side, compare Y-ranges. Merged line: [x,y1,x,y2]
                overlap = max(0, min(merged_line_segment[3], max(p1_qc[1], p2_qc[1])) - \
                                   max(merged_line_segment[1], min(p1_qc[1], p2_qc[1])))
            total_coverage_metric += (overlap / qc_side_len)

        coverage_score = (total_coverage_metric / num_valid_sides_for_coverage) if num_valid_sides_for_coverage > 0 else 0


        final_score = (w_edge_support * score +
                       w_parallelism * parallelism_score +
                       w_perpendicularity * perpendicularity_score +
                       w_length_similarity * length_similarity_score +
                       w_coverage * coverage_score
                       )

        print(f"Quad: {quad_corners.astype(int).tolist()}, ES:{score:.0f},Par:{parallelism_score:.2f},Perp:{perpendicularity_score:.2f},LenSim:{length_similarity_score:.2f},Cov:{coverage_score:.2f},Tot:{final_score:.2f}")

        return final_score


    def _get_candidate_edge_lines(self, merged_lines, is_horizontal, image_shape, num_candidates=3, min_length_ratio=0.1, is_primary_edge=True):
        """
        Selects a list of top N candidate lines for a given edge type.
        'is_primary_edge': True for top/left, False for bottom/right.
        'image_shape': (height, width) of the image for regional filtering.
        """
        if not merged_lines:
            return []

        img_h, img_w = image_shape

        # Determine length dimension and position dimension based on orientation
        length_dim_size = img_w if is_horizontal else img_h
        min_len_px = length_dim_size * min_length_ratio

        # Filter by minimum length
        if is_horizontal:
            long_lines = [l for l in merged_lines if abs(l[0] - l[2]) >= min_len_px]
        else: # Vertical
            long_lines = [l for l in merged_lines if abs(l[1] - l[3]) >= min_len_px]

        if not long_lines:
            return []

        # Regional filtering
        region_ratio_threshold = 0.4 # Consider lines in outer 40% of the image for each edge
        regional_candidates = []
        if is_horizontal:
            if is_primary_edge: # Top edge
                regional_candidates = [l for l in long_lines if (l[1] + l[3]) / 2 < img_h * region_ratio_threshold]
            else: # Bottom edge
                regional_candidates = [l for l in long_lines if (l[1] + l[3]) / 2 > img_h * (1 - region_ratio_threshold)]
        else: # Vertical
            if is_primary_edge: # Left edge
                regional_candidates = [l for l in long_lines if (l[0] + l[2]) / 2 < img_w * region_ratio_threshold]
            else: # Right edge
                regional_candidates = [l for l in long_lines if (l[0] + l[2]) / 2 > img_w * (1 - region_ratio_threshold)]

        if not regional_candidates:
             # Fallback if no lines in the strict region, use all long_lines
            regional_candidates = long_lines


        # Sort candidates: primary key is position (closer to edge is better), secondary is length (longer is better)
        if is_horizontal:
            # For top edge (is_primary_edge=True), sort Y ascending. For bottom, Y descending.
            regional_candidates.sort(key=lambda l: (((l[1] + l[3]) / 2), -abs(l[0] - l[2])), reverse=not is_primary_edge)
        else: # Vertical
            # For left edge (is_primary_edge=True), sort X ascending. For right, X descending.
            regional_candidates.sort(key=lambda l: (((l[0] + l[2]) / 2), -abs(l[1] - l[3])), reverse=not is_primary_edge)

        return regional_candidates[:num_candidates]


    def _select_dominant_lines(self, lines, is_horizontal, image_dim_size, y_cluster_tolerance=10, x_cluster_tolerance=10):
        """
        Selects two dominant lines (e.g., top/bottom or left/right) from a list of merged lines.
        'is_horizontal': True if lines are horizontal, False for vertical.
        'image_dim_size': Image height for H lines, image width for V lines.
        """
        # Default cluster tolerances passed as arguments, e.g., y_cluster_tolerance=10, x_cluster_tolerance=10
        if len(lines) < 2: # Need at least two lines to pick distinct top/bottom or left/right
            print(f"Not enough {'horizontal' if is_horizontal else 'vertical'} lines to select dominant pair ({len(lines)} found).")
            return None, None

        # 1. Cluster lines by position (Y for H-lines, X for V-lines)
        # Lines are assumed to be sorted by their primary axis position already by the caller
        # For H-lines: lines are [min_x, avg_y, max_x, avg_y]
        # For V-lines: lines are [avg_x, min_y, avg_x, max_y]

        clusters = []
        if not lines: return None, None

        current_cluster = [lines[0]]
        for i in range(1, len(lines)):
            line1 = current_cluster[-1] # Last line in current cluster
            line2 = lines[i]

            pos1 = (line1[1] + line1[3]) / 2 if is_horizontal else (line1[0] + line1[2]) / 2
            pos2 = (line2[1] + line2[3]) / 2 if is_horizontal else (line2[0] + line2[2]) / 2

            tolerance = y_cluster_tolerance if is_horizontal else x_cluster_tolerance
            if abs(pos2 - pos1) < tolerance:
                current_cluster.append(line2)
            else:
                clusters.append(list(current_cluster))
                current_cluster = [line2]
        clusters.append(list(current_cluster)) # Add the last cluster

        if len(clusters) < 2:
            print(f"Found only {len(clusters)} {'horizontal' if is_horizontal else 'vertical'} clusters. Need at least 2.")
            # Fallback: if only one cluster, maybe use its extreme lines if the cluster is thick?
            # Or, if it's a thin cluster, we can't pick two distinct lines.
            # For now, require >= 2 clusters.
            return None, None

        # 2. Score clusters (e.g., by total length of lines within them)
        scored_clusters = []
        for cluster in clusters:
            total_length = 0
            for line in cluster:
                length = abs(line[0] - line[2]) if is_horizontal else abs(line[1] - line[3])
                total_length += length
            # Store cluster, its average position, and its score
            avg_pos = np.mean([(l[1]+l[3])/2 if is_horizontal else (l[0]+l[2])/2 for l in cluster])
            scored_clusters.append({'lines': cluster, 'avg_pos': avg_pos, 'score': total_length})

        # Sort clusters by score (descending)
        # scored_clusters.sort(key=lambda c: c['score'], reverse=True)

        # We need to select two distinct clusters that are well-separated and likely card edges.
        # One approach: pick the two clusters with highest scores that are reasonably separated.
        # Another: pick one "low" (top/left) and one "high" (bottom/right) cluster.

        # Let's try picking the cluster closest to the start and the cluster closest to the end,
        # after filtering for some minimum score/quality.
        # For simplicity now, just take the first and last cluster from the position-sorted list of clusters.
        # This assumes the sorting of original lines + clustering preserves some order.

        # Clusters are already sorted by position because input lines were sorted.
        first_main_cluster = clusters[0]
        last_main_cluster = clusters[-1]

        def get_representative_line_from_cluster(cluster, is_horizontal_flag):
            if not cluster: return None

            if is_horizontal_flag:
                # For horizontal cluster:
                # Y position is the average of all lines' average Ys in the cluster.
                # X extent is min_x to max_x of all lines in the cluster.
                all_x_coords = []
                all_y_positions = []
                for line in cluster:
                    all_x_coords.extend([line[0], line[2]])
                    all_y_positions.append((line[1] + line[3]) / 2)
                if not all_x_coords: return None

                avg_y = int(round(np.mean(all_y_positions)))
                min_x = min(all_x_coords)
                max_x = max(all_x_coords)
                return [min_x, avg_y, max_x, avg_y]
            else:
                # For vertical cluster:
                # X position is the average of all lines' average Xs in the cluster.
                # Y extent is min_y to max_y of all lines in the cluster.
                all_y_coords = []
                all_x_positions = []
                for line in cluster:
                    all_y_coords.extend([line[1], line[3]])
                    all_x_positions.append((line[0] + line[2]) / 2)
                if not all_y_coords: return None

                avg_x = int(round(np.mean(all_x_positions)))
                min_y = min(all_y_coords)
                max_y = max(all_y_coords)
                return [avg_x, min_y, avg_x, max_y]

        line1 = get_representative_line_from_cluster(first_main_cluster, is_horizontal)
        line2 = get_representative_line_from_cluster(last_main_cluster, is_horizontal)

        if line1 is None or line2 is None:
            print(f"Could not get representative lines from clusters for {'horizontal' if is_horizontal else 'vertical'}.")
            return None, None

        # Ensure line1 is "before" line2 based on position
        pos1 = (line1[1] + line1[3]) / 2 if is_horizontal else (line1[0] + line1[2]) / 2
        pos2 = (line2[1] + line2[3]) / 2 if is_horizontal else (line2[0] + line2[2]) / 2
        if pos1 > pos2: # Swap if e.g. "top" line is below "bottom" line
            line1, line2 = line2, line1

        # print(f"Selected dominant {'H' if is_horizontal else 'V'} lines: {line1}, {line2}")
        return line1, line2


    def _merge_horizontal_lines(self, lines, y_tolerance=10, x_min_overlap_pixels=30):
        if not lines:
            return []

        # Sort lines by average y, then by x1
        # Lines are [x1, y1, x2, y2]
        lines.sort(key=lambda l: ((l[1] + l[3]) / 2, l[0]))

        merged_lines = []
        if not lines: return merged_lines

        current_merged_line = list(lines[0]) # Start with the first line as a potential merged line
                                            # current_merged_line stores [min_x, avg_y, max_x, avg_y_again]
                                            # but avg_y is calculated from all segments in the current group

        current_group_ys = [(lines[0][1] + lines[0][3]) / 2] # Store individual avg_y of lines in group
        current_group_y_coords = [lines[0][1], lines[0][3]] # Store all y coords for final avg calculation


        for i in range(1, len(lines)):
            curr_line = lines[i]

            # Average Y of the current line being considered
            curr_line_avg_y = (curr_line[1] + curr_line[3]) / 2

            # Average Y of the current_merged_line (based on lines grouped so far)
            # For comparison, use the avg_y of the *first* line in the current group, or overall group avg
            # Let's use the current_merged_line's running average Y.
            # current_merged_line[1] and [3] should be the same (the running average Y)

            # Check Y proximity with the current group's average Y
            group_avg_y = np.mean(current_group_y_coords)

            if abs(curr_line_avg_y - group_avg_y) < y_tolerance:
                # Check X overlap: if current_line's x-span overlaps with current_merged_line's x-span
                # current_merged_line: [min_x_group, group_avg_y, max_x_group, group_avg_y]
                group_min_x = current_merged_line[0]
                group_max_x = current_merged_line[2]

                # Overlap condition: one segment starts before the other ends, and they are somewhat close
                # A simple check: does curr_line's x-range connect to or overlap with group's x-range?
                # max(group_min_x, curr_line[0]) <= min(group_max_x, curr_line[2]) for any overlap
                # For merging, we might want them to be "close" even if not perfectly overlapping,
                # or a more direct overlap check:
                actual_overlap = max(0, min(group_max_x, curr_line[2]) - max(group_min_x, curr_line[0]))
                # Or, if one extends the other: (curr_line[0] <= group_max_x + x_min_overlap_pixels and curr_line[2] >= group_min_x - x_min_overlap_pixels)

                # If curr_line's x-range is "close enough" to the current merged group's x-range
                # "close enough" could mean they overlap, or one starts near where the other ends.
                # Let's use a simpler condition: if their x-spans are continuous or overlapping
                # when projected onto the x-axis.
                # We need to expand the current_merged_line if they merge.

                # Check if curr_line can extend or overlap with the current_merged_line
                # The segments must overlap in X or be connectable (within a small gap)
                # For simplicity, let's require direct overlap for now
                if actual_overlap >= x_min_overlap_pixels or \
                   (curr_line[0] >= group_min_x and curr_line[0] <= group_max_x + x_min_overlap_pixels) or \
                   (curr_line[2] <= group_max_x and curr_line[2] >= group_min_x - x_min_overlap_pixels):


                    current_merged_line[0] = min(current_merged_line[0], curr_line[0]) # new min_x
                    current_merged_line[2] = max(current_merged_line[2], curr_line[2]) # new max_x
                    current_group_y_coords.extend([curr_line[1], curr_line[3]])
                    new_avg_y = int(round(np.mean(current_group_y_coords)))
                    current_merged_line[1] = new_avg_y
                    current_merged_line[3] = new_avg_y
                else: # No X-overlap or not connectable, finalize previous merged_line
                    merged_lines.append(list(current_merged_line))
                    current_merged_line = list(curr_line)
                    current_group_y_coords = [curr_line[1], curr_line[3]]
                    avg_y = int(round(np.mean(current_group_y_coords)))
                    current_merged_line[1] = avg_y
                    current_merged_line[3] = avg_y
            else: # Y not close, finalize previous merged_line
                merged_lines.append(list(current_merged_line))
                current_merged_line = list(curr_line)
                current_group_y_coords = [curr_line[1], curr_line[3]]
                avg_y = int(round(np.mean(current_group_y_coords)))
                current_merged_line[1] = avg_y
                current_merged_line[3] = avg_y

        merged_lines.append(list(current_merged_line)) # Add the last merged line
        return merged_lines

    def _merge_vertical_lines(self, lines, x_tolerance=10, y_min_overlap_pixels=30):
        if not lines:
            return []

        # Sort lines by average x, then by y1
        lines.sort(key=lambda l: ((l[0] + l[2]) / 2, l[1]))

        merged_lines = []
        if not lines: return merged_lines

        current_merged_line = list(lines[0])
        current_group_x_coords = [lines[0][0], lines[0][2]]

        for i in range(1, len(lines)):
            curr_line = lines[i]
            curr_line_avg_x = (curr_line[0] + curr_line[2]) / 2
            group_avg_x = np.mean(current_group_x_coords)

            if abs(curr_line_avg_x - group_avg_x) < x_tolerance:
                group_min_y = current_merged_line[1]
                group_max_y = current_merged_line[3]

                actual_overlap = max(0, min(group_max_y, curr_line[3]) - max(group_min_y, curr_line[1]))

                if actual_overlap >= y_min_overlap_pixels or \
                   (curr_line[1] >= group_min_y and curr_line[1] <= group_max_y + y_min_overlap_pixels) or \
                   (curr_line[3] <= group_max_y and curr_line[3] >= group_min_y - y_min_overlap_pixels):

                    current_merged_line[1] = min(current_merged_line[1], curr_line[1]) # new min_y
                    current_merged_line[3] = max(current_merged_line[3], curr_line[3]) # new max_y
                    current_group_x_coords.extend([curr_line[0], curr_line[2]])
                    new_avg_x = int(round(np.mean(current_group_x_coords)))
                    current_merged_line[0] = new_avg_x
                    current_merged_line[2] = new_avg_x
                else:
                    merged_lines.append(list(current_merged_line))
                    current_merged_line = list(curr_line)
                    current_group_x_coords = [curr_line[0], curr_line[2]]
                    avg_x = int(round(np.mean(current_group_x_coords)))
                    current_merged_line[0] = avg_x
                    current_merged_line[2] = avg_x
            else:
                merged_lines.append(list(current_merged_line))
                current_merged_line = list(curr_line)
                current_group_x_coords = [curr_line[0], curr_line[2]]
                avg_x = int(round(np.mean(current_group_x_coords)))
                current_merged_line[0] = avg_x
                current_merged_line[2] = avg_x

        merged_lines.append(list(current_merged_line))
        return merged_lines


    def _get_line_intersection(self, line1, line2):
        """ Helper function to find intersection of two lines (x1,y1,x2,y2) format """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Denominator
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None # Parallel or coincident lines

        # Numerators
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

        # t and u are parameters for the line equations p = p0 + t*d
        t = t_num / den
        u = u_num / den

        # If 0 <= t <= 1 and 0 <= u <= 1, the intersection lies on both segments.
        # For finding corners of a rectangle from detected edges, we often want the
        # intersection of the lines themselves, not just the segments.
        # So, we calculate the intersection point regardless of t and u bounds here.
        # The calling function will need to validate if these points make sense.
        # if 0 <= t <= 1 and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return int(round(intersect_x)), int(round(intersect_y))
        # return None # This would be for segment-only intersection


    def find_card_edges(self, image_path):
        original_image, preprocessed_image = self._load_and_preprocess_image(image_path)

        # --- Attempt Hough Line Transform method ---
        hough_points = self._find_card_edges_hough(original_image, preprocessed_image)
        if hough_points is not None:
            # print("Using Hough method result.")
            ordered_points = self._order_points(hough_points)
            return original_image, ordered_points
        # else:
            # print("Hough method failed, falling back to contour method or returning None.")
            # Fallback or return None if Hough fails and contour method is removed/also fails
            # For now, let's just try Hough and if it fails, it means no detection.
            # This means we are replacing the contour method for now.

        # If Hough method does not return points, it means no card detected by this method.
        return original_image, None


        # --- Original Contour-based method (commented out for now) ---
        # edges_map = cv2.Canny(preprocessed_image, self.canny_threshold1, self.canny_threshold2)
        # kernel_morph = np.ones((5,5),np.uint8) # Back to 5x5 kernel. Removed OPEN.
        # closed_edges_map = cv2.morphologyEx(edges_map, cv2.MORPH_CLOSE, kernel_morph)
        # contours, _ = cv2.findContours(closed_edges_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # if not contours:
        #     return original_image, None

        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # image_area = original_image.shape[0] * original_image.shape[1]
        # min_area = image_area * self.min_contour_area_factor
        # candidate_contours = []
        # for contour in contours:
        #     if cv2.contourArea(contour) < min_area:
        #         break
        #     perimeter = cv2.arcLength(contour, True)
        #     approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) # Standard epsilon
        #     if len(approx) == 4:
        #         x, y, w, h = cv2.boundingRect(approx)
        #         aspect_ratio = float(w) / h
        #         is_valid_aspect_ratio = (self.expected_aspect_ratio_range[0] <= aspect_ratio <= self.expected_aspect_ratio_range[1]) or \
        #                                 (self.expected_aspect_ratio_range_inv[0] <= aspect_ratio <= self.expected_aspect_ratio_range_inv[1])
        #         if is_valid_aspect_ratio:
        #             candidate_contours.append(approx)

        # selected_contour = None
        # if not candidate_contours:
        #     return original_image, None

        # image_total_area = original_image.shape[0] * original_image.shape[1]
        # max_allowed_area_ratio = 0.95
        # if len(candidate_contours) > 1:
        #     valid_candidates_after_size_filter = []
        #     for c in candidate_contours:
        #         if cv2.contourArea(c) / image_total_area < max_allowed_area_ratio:
        #             valid_candidates_after_size_filter.append(c)

        #     if valid_candidates_after_size_filter:
        #         selected_contour = max(valid_candidates_after_size_filter, key=cv2.contourArea)
        #     else:
        #         selected_contour = min(candidate_contours, key=cv2.contourArea)
        # else:
        #     selected_contour = candidate_contours[0]

        # if selected_contour is not None:
        #     points = selected_contour.reshape(4, 2)
        #     ordered_points = self._order_points(points)
        #     return original_image, ordered_points
        # else:
        #     return original_image, None

    def _order_points(self, pts):
        # initialzie a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    def draw_edges(self, image, ordered_points, color=(0, 255, 0), thickness=2):
        """Draws the detected edges (a polygon from ordered points) on the image."""
        if ordered_points is None or len(ordered_points) != 4:
            return image # Return original image if no valid points

        # Ensure points are integers for drawing
        pts = np.array(ordered_points, dtype=np.int32)

        # Draw lines between the points
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), color, thickness) # TL to TR
        cv2.line(image, tuple(pts[1]), tuple(pts[2]), color, thickness) # TR to BR
        cv2.line(image, tuple(pts[2]), tuple(pts[3]), color, thickness) # BR to BL
        cv2.line(image, tuple(pts[3]), tuple(pts[0]), color, thickness) # BL to TL

        # Optionally, draw circles at corners
        for point in pts:
            cv2.circle(image, tuple(point), thickness * 2, color, -1)

        return image

if __name__ == '__main__':
    # Example usage (for testing purposes)
    # detector = CardDetector()
    # try:
    #     # Replace with an actual image path from the data/ directory for testing
    #     # edges = detector.find_card_edges("data/gpk-16a.webp")
    #     print("CardDetector initialized. Preprocessing step would run if an image path was provided.")
    # except FileNotFoundError as e:
    #     print(e)
    print("Card Detector module")
