import cv2
import numpy as np

def make_points(image, line):
    if line is None:
        return None
    slope, intercept = line
    y1 = image.shape[0]  # bottom of the image
    y2 = int(y1 * 3 / 5)  # slightly lower than the middle
    if slope == 0:  # to avoid division by zero
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    if lines is None:
        return None
    left_fit = []
    right_fit = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # vertical line
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:  # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    left_fit_average = np.average(left_fit, axis=0) if left_fit else None
    right_fit_average = np.average(right_fit, axis=0) if right_fit else None
    left_line = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    averaged_lines = [left_line, right_line]
    return [line for line in averaged_lines if line is not None]

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            if line is None:
                continue
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def region_of_interest(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    # Generalizing the ROI to a trapezoid for any road layout
    polygon = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.9), height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def detect_lines(image):
    canny_image = canny(image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(
        cropped_image, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        lines=np.array([]), 
        minLineLength=40, 
        maxLineGap=5
    )
    averaged_lines = average_slope_intercept(image, lines)
    line_image = display_lines(image, averaged_lines)
    combo_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    combo_image = detect_lines(frame)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()