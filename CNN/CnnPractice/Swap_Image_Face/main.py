

import cv2
import dlib
import numpy as np

# Load dlib's face detector and facial landmark predictor
frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("dataset/shape_predictor_68_face_landmarks.dat")

# Read source and target images
source_image = cv2.imread("dataset/source.jpg")
target_image = cv2.imread("dataset/target_face.jpg")

# Convert images to grayscale
source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
target_image_grayscale = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

# Display the grayscale images
# cv2.imshow("source_image", source_image_grayscale)
# cv2.imshow("target_image", target_image_grayscale)

# Creating zero array canvas for source and destination images
source_image_canvas = np.zeros_like(source_image_grayscale)
height, width, no_of_channels = target_image.shape
target_image_canvas = np.zeros((height, width, no_of_channels), np.uint8)

# Function to get the index from a numpy array
def index_from_array(numpyarray):
    index = None
    for n in numpyarray[0]:
        index = n
        break
    return index

# Find faces in the source image
source_faces = frontal_face_detector(source_image_grayscale)
for source_face in source_faces:
    source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
    source_face_landmark_points = []

    for landmark_no in range(0, 68):
        x_point = source_face_landmarks.part(landmark_no).x
        y_point = source_face_landmarks.part(landmark_no).y
        source_face_landmark_points.append((x_point, y_point))

    # Find convex hull of the source face
    source_face_landmark_points_array = np.array(source_face_landmark_points, np.int32)
    source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)

    # Fill the convex hull to create a mask
    cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)

    # Place the mask over the source image
    source_face_image = cv2.bitwise_and(source_image, source_image, mask=source_image_canvas)

    # Finding Delaunay triangulation indices of the source image
    bounding_rectangle = cv2.boundingRect(source_face_convexhull)
    subdivisions = cv2.Subdiv2D(bounding_rectangle)
    subdivisions.insert(source_face_landmark_points)
    triangles_vector = subdivisions.getTriangleList()
    triangles_array = np.array(triangles_vector, dtype=np.int32)
    source_triangle_index_points_list = []

    for triangle in triangles_array:
        index_point1 = (triangle[0], triangle[1])
        index_point2 = (triangle[2], triangle[3])
        index_point3 = (triangle[4], triangle[5])

        index_point1 = np.where((source_face_landmark_points_array == index_point1).all(axis=1))
        index_point1 = index_from_array(index_point1)

        index_point2 = np.where((source_face_landmark_points_array == index_point2).all(axis=1))
        index_point2 = index_from_array(index_point2)

        index_point3 = np.where((source_face_landmark_points_array == index_point3).all(axis=1))
        index_point3 = index_from_array(index_point3)

        triangle = [index_point1, index_point2, index_point3]
        source_triangle_index_points_list.append(triangle)

# Process the target image
target_faces = frontal_face_detector(target_image_grayscale)
for target_face in target_faces:
    target_face_landmarks = frontal_face_predictor(target_image_grayscale, target_face)
    target_face_landmark_points = []

    for landmark_no in range(0, 68):
        x_point = target_face_landmarks.part(landmark_no).x
        y_point = target_face_landmarks.part(landmark_no).y
        target_face_landmark_points.append((x_point, y_point))

    # Find convex hull of the target face
    target_face_landmark_points_array = np.array(target_face_landmark_points, np.int32)
    target_face_convexhull = cv2.convexHull(target_face_landmark_points_array)

    # Process each triangle from the source image
    for i, triangle_index_points in enumerate(source_triangle_index_points_list):
        source_triangle_point1 = source_face_landmark_points[triangle_index_points[0]]
        source_triangle_point2 = source_face_landmark_points[triangle_index_points[1]]
        source_triangle_point3 = source_face_landmark_points[triangle_index_points[2]]
        source_triangle = np.array([source_triangle_point1, source_triangle_point2, source_triangle_point3], np.int32)
        source_rectangle = cv2.boundingRect(source_triangle)
        (x, y, w, h) = source_rectangle
        cropped_source_rectangle = source_image[y:y+h, x:x+w]
        source_triangle_points = np.array([[source_triangle_point1[0] - x, source_triangle_point1[1] - y],
                                           [source_triangle_point2[0] - x, source_triangle_point2[1] - y],
                                           [source_triangle_point3[0] - x, source_triangle_point3[1] - y]], np.int32)

        # For the target image
        target_triangle_point1 = target_face_landmark_points[triangle_index_points[0]]
        target_triangle_point2 = target_face_landmark_points[triangle_index_points[1]]
        target_triangle_point3 = target_face_landmark_points[triangle_index_points[2]]
        target_triangle = np.array([target_triangle_point1, target_triangle_point2, target_triangle_point3], np.int32)
        target_rectangle = cv2.boundingRect(target_triangle)
        (x, y, w, h) = target_rectangle
        cropped_target_rectangle_mask = np.zeros((h, w), np.uint8)
        target_triangle_points = np.array([[target_triangle_point1[0] - x, target_triangle_point1[1] - y],
                                           [target_triangle_point2[0] - x, target_triangle_point2[1] - y],
                                           [target_triangle_point3[0] - x, target_triangle_point3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_target_rectangle_mask, target_triangle_points, 255)

        # Warp the source triangle to match the target
        source_triangle_points = np.float32(source_triangle_points)
        target_triangle_points = np.float32(target_triangle_points)
        matrix = cv2.getAffineTransform(source_triangle_points, target_triangle_points)
        warped_triangle = cv2.warpAffine(cropped_source_rectangle, matrix, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_target_rectangle_mask)

        # Reconstruct the destination face in an empty canvas
        new_dest_face_canvas_area = target_image_canvas[y:y+h, x:x+w]
        new_dest_face_canvas_area_gray = cv2.cvtColor(new_dest_face_canvas_area, cv2.COLOR_BGR2GRAY)
        _, mask_created_triangle = cv2.threshold(new_dest_face_canvas_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_created_triangle)
        new_dest_face_canvas_area = cv2.add(new_dest_face_canvas_area, warped_triangle)
        target_image_canvas[y:y+h, x:x+w] = new_dest_face_canvas_area

# Swapping the target face with the newly created face
final_target_canvas = np.zeros_like(target_image_grayscale)
final_target_face_mask = cv2.fillConvexPoly(final_target_canvas, target_face_convexhull, 255)
final_target_canvas = cv2.bitwise_not(final_target_face_mask)
target_face_masked = cv2.bitwise_and(target_image, target_image, mask=final_target_canvas)
target_with_face = cv2.add(target_face_masked, target_image_canvas)

# Seamless clone to make the attachment blend with the surrounding pixels
(x, y, w, h) = cv2.boundingRect(target_face_convexhull)
target_face_center_point = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlesscloned_face = cv2.seamlessClone(target_with_face, target_image, final_target_face_mask, target_face_center_point, cv2.NORMAL_CLONE)

# Display the final result
# cv2.imshow("Final Swapped Face", seamlesscloned_face)
cv2.imwrite("final_seamlesscloned_face.jpg", seamlesscloned_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
