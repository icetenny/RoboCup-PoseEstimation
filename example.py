from pose_estimation import PoseEstimation

import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# Detection = initial detection, Tracking = Tracking after initial detection
with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        start = time.time()
        PE = PoseEstimation()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            image = frame.copy()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # pose and hands detection
            pose_results = pose.process(image)  # Detection
            hands_results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # init frame each loop
            PE.read_results(cap, image, pose_results, hands_results)

            # Rendering results
            if pose_results.pose_landmarks:
                PE.draw_pose()
                PE.detect_hand_raise()
                # min_distance = 1/20 width of mouth
                PE.detect_nod(time_interval=0.5,
                              min_distance=
                              PE.get_distance(PE.get_exact_pose_coords(9), PE.get_exact_pose_coords(10))[
                                  -1] / 20, draw_line=True)

            # box_list = [[name, (x,y,w,h), is_pointed], ...]
            box_list = [["A", (100, 100, 30, 40), False], ["B", (150, 250, 60, 60), False],
                        ["C", (400, 100, 20, 25), False], ["D", (500, 400, 40, 40), False],
                        ["E", (150, 400, 50, 60), False]]

            # finger_list = [(startindex, midindex, length), ...]
            finger_list = [(7, 8, 200)]

            if hands_results.multi_hand_landmarks:
                PE.draw_hand()
                PE.draw_hand_label()
                PE.point_to(box_list, finger_list)

            PE.draw_boxes(box_list)

            # get fps
            fps = 1 / (time.time() - start)
            start = time.time()
            cv2.putText(image, "fps: " + str(round(fps, 2)), (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

            cv2.imshow("Original", frame)
            cv2.imshow("image", image)

            if cv2.waitKey(5) == ord("q"):
                cap.release()
cv2.destroyAllWindows()
