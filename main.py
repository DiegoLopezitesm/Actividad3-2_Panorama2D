import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Config
CHESSBOARD_SIZE = (7, 5)
SQUARE_SIZE_MM = 25.0

CALIB_DIR = "calib"
PANO_DIR = "images"
OUTPUT_DIR = "output"
CALIB_FILE = os.path.join(OUTPUT_DIR, "camera_params.npz")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calibrate_camera():
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points, img_points = [], []

    images = sorted(
        glob.glob(os.path.join(CALIB_DIR, "*.jpg")) +
        glob.glob(os.path.join(CALIB_DIR, "*.png")) +
        glob.glob(os.path.join(CALIB_DIR, "*.jpeg"))
    )

    if not images:
        raise FileNotFoundError("No hay imágenes de calibración.")

    print(f"[CALIB] {len(images)} imágenes")

    img_shape = None

    for f in images:
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)

        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners)
            print("✔", os.path.basename(f))
        else:
            print("✘", os.path.basename(f))

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None,
        flags=cv2.CALIB_FIX_K3
    )

    print("[CALIB] RMS:", rms)
    np.savez(CALIB_FILE, K=K, dist=dist)

    return K, dist


def load_camera():
    data = np.load(CALIB_FILE)
    return data["K"], data["dist"]


def undistort(img, K, dist):
    h, w = img.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    return cv2.undistort(img, K, dist, None, newK)


def sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def match(descA, descB):
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=50)
    )

    matches = flann.knnMatch(descA, descB, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good


def homography(kpA, kpB, matches):
    ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 5.0)
    return H, mask


def stitch(a, b, H):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]

    ca = np.float32([[0,0],[wa,0],[wa,ha],[0,ha]]).reshape(-1,1,2)
    cb = np.float32([[0,0],[wb,0],[wb,hb],[0,hb]]).reshape(-1,1,2)

    cb = cv2.perspectiveTransform(cb, H)

    allc = np.concatenate((ca, cb), axis=0)
    xmin, ymin = np.int32(allc.min(axis=0).ravel())
    xmax, ymax = np.int32(allc.max(axis=0).ravel())

    t = [-xmin, -ymin]
    T = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]], dtype=np.float32)

    out = cv2.warpPerspective(b, T @ H, (xmax-xmin, ymax-ymin))
    out[t[1]:t[1]+ha, t[0]:t[0]+wa] = a

    return out


def build_panorama(K=None, dist=None):
    paths = sorted(glob.glob(os.path.join(PANO_DIR, "*.*")))
    imgs = [cv2.imread(p) for p in paths]

    if K is not None:
        imgs = [undistort(i, K, dist) for i in imgs]

    pano = imgs[0]

    for i in range(1, len(imgs)):
        A, B = pano, imgs[i]

        kpA, descA = sift(A)
        kpB, descB = sift(B)

        good = match(descA, descB)

        if len(good) < 4:
            continue

        H, _ = homography(kpA, kpB, good)

        if H is None:
            continue

        pano = stitch(A, B, H)

    return pano


def main():
    if os.path.exists(CALIB_FILE):
        K, dist = load_camera()
    else:
        K, dist = calibrate_camera()

    pano = build_panorama(K, dist)

    out = os.path.join(OUTPUT_DIR, "panorama.jpg")
    cv2.imwrite(out, pano)

    plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print("Guardado en:", out)


if __name__ == "__main__":
    main()