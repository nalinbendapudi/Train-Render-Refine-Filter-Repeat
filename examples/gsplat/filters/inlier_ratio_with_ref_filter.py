import cv2
import numpy as np


def extract_features(image, method="sift"):
    """
    Detect keypoints + descriptors using SIFT or ORB
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "sift":
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(5000)
        norm_type = cv2.NORM_HAMMING

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors, norm_type


def match_descriptors(desc1, desc2, norm_type):
    """
    Match descriptors using Lowe ratio test
    """
    if desc1 is None or desc2 is None:
        return []

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches


def compute_inlier_ratio(query_img, reference_img, method="sift"):
    """
    Compute geometric verification quality using:
    feature matching + Essential Matrix RANSAC

    Returns:
        inlier_ratio
        num_matches
        num_inliers
    """

    kp1, desc1, norm_type = extract_features(query_img, method)
    kp2, desc2, _ = extract_features(reference_img, method)

    matches = match_descriptors(desc1, desc2, norm_type)

    if len(matches) < 8:
        return {
            "success": False,
            "inlier_ratio": 0.0,
            "num_matches": len(matches),
            "num_inliers": 0
        }

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=1.0,
        confidence=0.999
    )

    if F is None or mask is None:
        return {
            "success": False,
            "inlier_ratio": 0.0,
            "num_matches": len(matches),
            "num_inliers": 0
        }

    num_inliers = int(mask.sum())
    inlier_ratio = num_inliers / len(matches)

    return {
        "success": True,
        "inlier_ratio": inlier_ratio,
        "num_matches": len(matches),
        "num_inliers": num_inliers
    }


def should_discard_refined(rendered_img, refined_img, reference_img):
    """
    Rule:
    If refined inlier ratio < rendered inlier ratio
    -> discard refined
    """

    rendered_result = compute_inlier_ratio(
        rendered_img,
        reference_img
    )

    refined_result = compute_inlier_ratio(
        refined_img,
        reference_img
    )

    print("\nRendered Result:")
    print(rendered_result)

    print("\nRefined Result:")
    print(refined_result)

    if not refined_result["success"]:
        print("\nRefined failed geometric verification -> discard")
        return True

    if refined_result["inlier_ratio"] < rendered_result["inlier_ratio"]:
        print("\nDiscard refined: lower inlier ratio")
        return True

    print("\nKeep refined view")
    return False


if __name__ == "__main__":
    rendered_img = cv2.imread("rendered.png")
    refined_img = cv2.imread("refined.png")
    reference_img = cv2.imread("reference.png")

    discard = should_discard_refined(
        rendered_img,
        refined_img,
        reference_img
    )

    print("\nDiscard refined?", discard)