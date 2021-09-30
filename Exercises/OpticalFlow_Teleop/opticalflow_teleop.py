from GUI import GUI
from HAL import HAL
import cv2 as cv
import numpy as np
from scipy.stats import mode



# Parameter initialization
size_accumulator = 10
directions_map = np.zeros([size_accumulator, 5])

frame_previous = HAL.getImage()
gray_previous = cv.cvtColor(frame_previous, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame_previous)
hsv[:, :, 1] = 255
threshold = 10.0, # Threshold value for magnitude

param = {
    'pyr_scale': 0.5, # Image scale (<1) to build pyramids for each image
    'levels': 3, # Number of pyramid layers
    'winsize': 15, # Averaging window size
    'iterations': 3, # Number of iterations the algorithm does at each pyramid level
    'poly_n': 5, # Size of the pixel neighborhood used to find polynomial expansion in each pixel
    'poly_sigma': 1.1, # Standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion
    'flags': cv.OPTFLOW_LK_GET_MIN_EIGENVALS
}

while True:

    frame = HAL.getImage()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(gray_previous, gray, None, **param)
    mag, ang = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    ang_180 = ang/2
    gray_previous = gray
        
    move_sense = ang[mag > threshold]
    move_mode = mode(move_sense)[0]

    if 10 < move_mode <= 100:
        directions_map[-1, 0] = 1
        directions_map[-1, 1:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 100 < move_mode <= 190:
        directions_map[-1, 1] = 1
        directions_map[-1, :1] = 0
        directions_map[-1, 2:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 190 < move_mode <= 280:
        directions_map[-1, 2] = 1
        directions_map[-1, :2] = 0
        directions_map[-1, 3:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 280 < move_mode or move_mode < 10:
        directions_map[-1, 3] = 1
        directions_map[-1, :3] = 0
        directions_map[-1, 4:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    else:
        directions_map[-1, -1] = 1
        directions_map[-1, :-1] = 0
        directions_map = np.roll(directions_map, 1, axis=0)


    loc = directions_map.mean(axis=0).argmax()
    if loc == 0:
        text = 'Back'
        HAL.motors.sendV(-1)
    elif loc == 1:
        text = 'Turn right'
        HAL.motors.sendW(-0.05)
    elif loc == 2:
        text = 'Advance'
        HAL.motors.sendV(1)
    elif loc == 3:
        text = 'Turn left'
        HAL.motors.sendW(0.05)
    else:
        text = 'WAITING'

    hsv[:, :, 0] = ang_180
    hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    frame = cv.flip(frame, 1)
    cv.putText(frame, text, (30, 90), cv.FONT_HERSHEY_COMPLEX, frame.shape[1] / 500, (0, 0, 255), 2)

    # Show Image
    GUI.showImage(frame)

cap.release()
cv.destroyAllWindows()