import cv2

from get_background import get_background

# cap = cv2.VideoCapture("http://192.168.45.20:8080/video")
# cap = cv2.VideoCapture("./input/video_1.mp4")
# cap = cv2.VideoCapture("./input/video_2.mp4")
# cap = cv2.VideoCapture("./input/video_3.mp4")
cap = cv2.VideoCapture("./input/video_4.mov")
# cap = cv2.VideoCapture("./input/video_6.mp4")
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

# get the background model
static_back = None

frame_count = 0
back_frames = []

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        # uncomment for camera and video_4
        frame = cv2.resize(frame, (450, 900))

        frame_count += 1

        orig_frame = frame.copy()

        if frame_count < 100:
            back_frames.append(frame.copy())
            continue

        if static_back is None:
            static_back = get_background(back_frames)
            cv2.imshow('static_back', static_back)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        cv2.imshow('blurred', blurred)

        # find the difference between current frame and base frame
        frame_diff = cv2.absdiff(static_back, blurred)
        cv2.imshow('frame_diff', frame_diff)

        # thresholding to convert the frame to binary
        ret, thres = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        cv2.imshow('thres', thres)

        # dilate the frame a bit to get some more white area...
        # ... makes the detection of contours a bit easier
        dilate_frame = cv2.dilate(thres, None, iterations=5)

        cv2.imshow('dilate_frame', dilate_frame)

        contours, hierarchy = cv2.findContours(
            dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
