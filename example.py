from pose_estimation import PoseEstimation
import mediapipe as mp
import cv2
import time

# last mod: 29/1/2022

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

# [name, (x, y, w, h)]
box_list = [["Apple", (100, 150, 30, 40)], ["Ball", (150, 250, 60, 60)],
            ["Cola", (400, 100, 20, 25)], ["Doraemon", (500, 400, 40, 40)],
            ["Egg", (150, 400, 50, 60)]]

# (start_hand_index, middle_hand_index, length)
finger_list = [(7, 8, 200)]
# see src for all hand landmarks index

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose, mp_hands.Hands(
        min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
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

        pose_results = pose.process(image)
        hands_results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        PE.read_results(cap, image, pose_results, hands_results)

        if pose_results.pose_landmarks:
            PE.draw_pose()

            raise_hand = PE.detect_hand_raise(print_result=False, screen_label=True)
            # if raise_hand:
            #     print(raise_hand)

            # time interval = max time you can stop moving before the program reset
            # min distance = min distance for the program to detect movement, in this case = 1/20 length of mouth
            # get_exact_pose_coords(pose_landmark_index) return dx, dy, dz, dxy : see src for all landmark index
            nod = PE.detect_nod(time_interval=0.3,
                                min_distance=
                                PE.get_distance(PE.get_exact_pose_coords(9), PE.get_exact_pose_coords(10))[
                                    -1] / 20, draw_line=True, print_result=False, screen_label=True)
            # if nod:
            #     print(nod)

        # draw green boxes for every object in box_list
        PE.draw_boxes(box_list)

        if hands_results.multi_hand_landmarks:
            PE.draw_hand()
            PE.draw_hand_label()

            pointed_box_list = PE.point_to(box_list, finger_list, print_result=False, screen_label=True)

            PE.draw_boxes(pointed_box_list, is_pointed=True)
            # if pointed_box_list:
            #     print(pointed_box_list)

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
