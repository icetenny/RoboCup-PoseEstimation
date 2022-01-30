import mediapipe as mp
import cv2
import numpy as np
import time
import math

# v4
# last mod: 30/1/2022 22:30

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles


class PoseEstimation:
    def __init__(self, min_pose_detect_conf=0.8, min_pose_track_conf=0.5, min_hands_detect_conf=0.8,
                 min_hand_track_conf=0.5, max_num_hands=2):

        self.pose = mp_pose.Pose(min_detection_confidence=min_pose_detect_conf,
                                 min_tracking_confidence=min_pose_track_conf)
        self.hands = mp_hands.Hands(min_detection_confidence=min_hands_detect_conf,
                                    min_tracking_confidence=min_hand_track_conf, max_num_hands=max_num_hands)

        self.last_moving_time = 0
        self.nose_coords_bfat = [(0, 0, 0), (0, 0, 0)]
        self.nose_moving_coords_list = []
        self.nose_vector_list = []

    def process_frame(self, frame):
        self.frame_height, self.frame_width = frame.shape[:-1]

        image = frame.copy()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.pose_results = self.pose.process(image)
        self.hands_results = self.hands.process(image)

        self.pose_detected = bool(self.pose_results.pose_landmarks)
        self.hands_detected = bool(self.hands_results.multi_hand_landmarks)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.image = image
        return self.image

    # POSE

    def get_pose_coords(self, landmark_index):
        if self.pose_detected:
            return tuple(np.multiply(
                np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                          self.pose_results.pose_landmarks.landmark[landmark_index].y,
                          self.pose_results.pose_landmarks.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]).astype(int))

    def get_exact_pose_coords(self, landmark_index):
        if self.pose_detected:
            return tuple(np.multiply(
                np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                          self.pose_results.pose_landmarks.landmark[landmark_index].y,
                          self.pose_results.pose_landmarks.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]))

    def get_pose_joint_angle(self, joint):
        if self.pose_detected:
            co1, co2, co3 = [self.get_exact_pose_coords(joint[i]) for i in range(3)]

            radxy = np.arctan2(co3[1] - co2[1], co3[0] - co2[0]) - np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            anglexy = np.abs(radxy * 180 / np.pi)
            anglexy = min(anglexy, 360 - anglexy)
            return anglexy

    def show_pose_joint_angles(self, image, joint_list):
        if self.pose_detected:
            for joint in joint_list:
                joint_angle = self.get_pose_joint_angle(joint)

                cv2.putText(image, str(round(joint_angle, 2)), self.get_pose_coords(joint[1])[:2],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)

            return image

    def get_pose_slope_angle(self, index1, index2):
        if self.pose_detected:
            co1, co2 = self.get_exact_pose_coords(index1), self.get_exact_pose_coords(index2)

            slope_radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            slope_anglexy = np.abs(slope_radxy * 180 / np.pi)

            return slope_anglexy

    def draw_pose(self):
        if self.pose_detected:
            mp_drawing.draw_landmarks(
                self.image,
                self.pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    def detect_hand_raise(self, print_result=False, screen_label=False):
        if self.pose_detected:
            raised_hand_list = []

            right_shoulder = self.get_exact_pose_coords(12)
            right_elbow = self.get_exact_pose_coords(14)
            right_wrist = self.get_exact_pose_coords(16)

            if right_wrist[1] <= right_shoulder[1] and \
                    right_wrist[1] <= right_elbow[1] and \
                    (120 >= self.get_pose_joint_angle((12, 14, 16)) >= 30 or
                     135 >= self.get_pose_slope_angle(14, 16) >= 45) and 10 <= self.get_pose_slope_angle(14, 16) <= 170:
                if print_result:
                    print("Right hand raised", self.get_pose_coords(16)[:-1])
                raised_hand_list.append(("R", right_wrist[:-1]))

            left_shoulder = self.get_exact_pose_coords(11)
            left_elbow = self.get_exact_pose_coords(13)
            left_wrist = self.get_exact_pose_coords(15)

            if left_wrist[1] <= left_shoulder[1] and \
                    left_wrist[1] <= left_elbow[1] and \
                    (120 >= self.get_pose_joint_angle((11, 13, 15)) >= 30 or
                     135 >= self.get_pose_slope_angle(13, 15) >= 45) and 10 <= self.get_pose_slope_angle(13, 15) <= 170:
                if print_result:
                    print("Left hand raised", self.get_pose_coords(15)[:-1])
                raised_hand_list.append(("L", left_wrist[:-1]))

            if raised_hand_list and screen_label:
                cv2.putText(self.image,
                            " ".join(
                                [{"R": "Right hand raised", "L": "Left hand raised"}[h[0]] for h in raised_hand_list]),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            return raised_hand_list

    def get_distance(self, p1, p2):
        if self.pose_detected or self.hands_detected:
            dx, dy, dz = (p2[i] - p1[i] for i in range(3))
            dxy = (dx ** 2 + dy ** 2) ** 0.5

            return dx, dy, dz, dxy

    def is_moving(self, coords_bfat, time_interval, min_distance):
        if self.pose_detected:
            dx, dy, dz, dxy = self.get_distance(*coords_bfat)
            if dxy > min_distance:
                self.last_moving_time = time.time()
                if abs(dx) > abs(dy):
                    if dx >= 0:
                        return "x"
                    else:
                        return "-x"
                else:
                    if dy >= 0:
                        return "y"
                    else:
                        return "-y"
            elif time.time() - self.last_moving_time < time_interval:
                return "o"
            else:
                return False

    def draw_moving_line(self, image, landmark_index, coords_list, is_moving, vector_list, draw_line, print_result,
                         color=(255, 0, 0), thickness=3):
        if self.pose_detected:
            if is_moving:
                if is_moving in "-x-y" and [is_moving] != vector_list[-1:]:
                    vector_list.append(is_moving)
                coords_list.append(self.get_pose_coords(landmark_index)[:2])

                if draw_line:
                    for start_point, end_point in zip(coords_list, coords_list[1:]):
                        cv2.line(image, start_point, end_point, color, thickness)

            elif vector_list:
                coords_list.clear()
                vector_list.clear()
                if print_result:
                    print("Head stopped moving")

    def detect_nod(self, time_interval=0.5, min_distance=2, draw_line=True, print_result=False, screen_label=False):
        if self.pose_detected:
            self.nose_coords_bfat = [self.nose_coords_bfat[1], self.get_exact_pose_coords(0)]

            self.draw_moving_line(self.image, 0, self.nose_moving_coords_list,
                                  self.is_moving(self.nose_coords_bfat, time_interval=time_interval,
                                                 min_distance=min_distance), self.nose_vector_list, draw_line,
                                  print_result)

            # latest 15 vectors
            lv = self.nose_vector_list[-15:]

            if lv.count("x") > 0 and lv.count("-x") > 0 and lv.count("x") + lv.count("-x") >= 0.7 * len(lv):
                if screen_label:
                    cv2.putText(self.image, "Shaking", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if print_result:
                    print("Shaking detected")
                return "Shaking"
            if lv.count("y") > 0 and lv.count("-y") > 0 and lv.count("y") + lv.count("-y") >= 0.7 * len(lv):
                if screen_label:
                    cv2.putText(self.image, "Nodding", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if print_result:
                    print("Nodding detected")
                return "Nodding"

            return None

    # HAND

    def get_hand_coords(self, hand, landmark_index):
        if self.hands_detected:
            return tuple(np.multiply(
                np.array(
                    (
                        hand.landmark[landmark_index].x, hand.landmark[landmark_index].y,
                        hand.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]).astype(int))

    def get_exact_hand_coords(self, hand, landmark_index):
        if self.hands_detected:
            return tuple(np.multiply(
                np.array(
                    (
                        hand.landmark[landmark_index].x, hand.landmark[landmark_index].y,
                        hand.landmark[landmark_index].z)),
                [self.frame_width, self.frame_height, self.frame_width]))

    def get_hand_label(self, index, hand, results):
        if self.hands_detected:
            classification = results.multi_handedness[index]
            label = classification.classification[0].label
            label = ("Right", "Left")[("Left", "Right").index(label)]
            score = classification.classification[0].score
            txt = "{} {}".format(label, round(score, 2))
            coords = self.get_hand_coords(hand, 0)[:2]

            return txt, coords

    def draw_finger_angles(self, image, hand, joint_list):
        if self.hands_detected:
            for joint in joint_list:
                co1, co2, co3 = [self.get_hand_coords(hand, joint[i]) for i in range(3)]
                print(co1, co2, co3)

                radxy = np.arctan2(co3[1] - co2[1], co3[0] - co2[0]) - np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
                anglexy = np.abs(radxy * 180 / np.pi)
                anglexy = min(anglexy, 360 - anglexy)

                cv2.putText(image, str(round(anglexy, 2)), co2[:2], cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2, cv2.LINE_AA)

            return image

    def get_hand_slope_angle(self, hand, index1, index2):
        if self.hands_detected:
            co1, co2 = self.get_exact_hand_coords(hand, index1), self.get_exact_hand_coords(hand, index2)

            radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
            return radxy

    def get_hand_slope(self, hand, index1, index2):
        if self.hands_detected:
            co1, co2 = self.get_exact_hand_coords(hand, index1), self.get_exact_hand_coords(hand, index2)
            slope = (co2[1] - co1[1]) / (co2[0] - co1[0])

            return slope

    def draw_cont_line(self, hand, image, start_point, mid_point, length=200, color=(0, 255, 0), thickness=2):
        if self.hands_detected:
            co_mid = self.get_hand_coords(hand, mid_point)
            co_start = self.get_hand_coords(hand, start_point)
            slope = self.get_hand_slope(hand, start_point, mid_point)
            slope_angle = self.get_hand_slope_angle(hand, start_point, mid_point)

            if co_mid[0] >= co_start[0]:
                xlen = round(abs(math.cos(slope_angle) * length))
            else:
                xlen = -round(abs(math.cos(slope_angle) * length))

            if co_mid[1] >= co_start[1]:
                ylen = round(abs(math.sin(slope_angle) * length))
            else:
                ylen = -round(abs(math.sin(slope_angle) * length))

            cv2.line(image, co_mid[:2], (co_mid[0] + xlen, co_mid[1] + ylen), color, thickness)

            return co_start, co_mid, slope

    def draw_hand(self):
        if self.hands_detected:
            for num, hand in enumerate(self.hands_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(self.image, hand, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())

    def draw_box(self, image, box_name, xywh_tuple, is_pointed):
        x, y, w, h = xywh_tuple
        if is_pointed:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, box_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    def draw_boxes(self, box_list, is_pointed=False):
        for box in box_list:
            self.draw_box(self.image, *box, is_pointed)

    def draw_hand_label(self):
        if self.hands_detected:
            for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

                if self.get_hand_label(num, hand, self.hands_results):
                    text, coord = self.get_hand_label(num, hand, self.hands_results)
                    cv2.putText(self.image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def point_to(self, box_list, finger_list, print_result=False, screen_label=False):
        if self.hands_detected:
            pointed_box_list = []

            for box in box_list:
                box_name, xywh = box
                bx, by, bw, bh = xywh
                is_pointed = False

                for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

                    for finger in finger_list:
                        co_start, co_mid, slope = self.draw_cont_line(hand, self.image, *finger, color=(255, 0, 0))
                        finger_len = finger[2]

                        # y-intercept
                        c = co_mid[1] - slope * co_mid[0]

                        # get range of x and y
                        if co_start[0] >= co_mid[0]:
                            range_x = [0, co_mid[0]]
                        else:
                            range_x = [co_mid[0] + 1, self.frame_width]

                        if co_start[1] >= co_mid[1]:
                            range_y = [0, co_mid[1]]
                        else:
                            range_y = [co_mid[1] + 1, self.frame_height]

                        # if box in range x and y
                        if (range_x[0] <= bx <= range_x[1] or range_x[0] <= bx + bw <= range_x[1]) \
                                and (range_y[0] <= by <= range_y[1] or range_y[0] <= by + bh <= range_y[1]):
                            y_bx = slope * bx + c
                            y_bxw = slope * (bx + bw) + c

                            # if not line goes above or below box
                            if not ((y_bx < by and y_bxw < by) or (
                                    y_bx > by + bh and y_bxw > by + bh)) and \
                                    finger_len >= self.get_distance(co_mid, (bx + bw / 2, by + bh / 2, 0))[-1] - (
                                    bw + bh) / 2:
                                is_pointed = True
                                pointed_box_list.append(box)
                                break
                    if is_pointed:
                        break

            if pointed_box_list:
                if screen_label:
                    cv2.putText(self.image, "Pointed at: " + ",".join([b[0] for b in pointed_box_list if b[-1]]),
                                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if print_result:
                    print("Pointed at {}".format([b[0] for b in box_list if b[-1]]))

            return pointed_box_list
