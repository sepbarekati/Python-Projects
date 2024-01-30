from time import sleep
import cv2

from get_background import get_background

# cap = cv2.VideoCapture("http://192.168.1.155:8080/video")
# cap = cv2.VideoCapture("./input/video_1.mp4")
# cap = cv2.VideoCapture("./input/video_2.mp4")
# cap = cv2.VideoCapture("./input/video_3.mp4")
cap = cv2.VideoCapture("./input/video_4.mov")
# get the video frame height and width
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = "output/out.mp4"
# define codec and create VideoWriter object
out = cv2.VideoWriter(
    save_name,
    cv2.VideoWriter_fourcc(*'mp4v'), 10,
    (frame_width, frame_height)
)


background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC()

frame_count = 0
back_frames = []

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.resize(frame, (450, 900))
        frame_count += 1

        orig_frame = frame.copy()

        if frame_count < 100:
            back_frames.append(frame.copy())
            continue

        # frame = cv2.resize(frame, (640, 360))
        foreground_mask = background_subtr_method.apply(frame)
        background_img = background_subtr_method.getBackgroundImage()

        cv2.imshow('foreground_mask', foreground_mask)
        cv2.imshow('background_img', background_img)

        contours, hierarchy = cv2.findContours(
            foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 500:
                continue

            cv2.drawContours(orig_frame, contours, i, (0, 0, 255), 3)

        cv2.imshow('Detected Objects', orig_frame)
        out.write(orig_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
