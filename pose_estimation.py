import mediapipe as mp
import cv2
import numpy as np
import time
import math


class PoseEstimation:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.last_moving_time = 0
        self.nose_coords_bfat = [(0, 0, 0), (0, 0, 0)]
        self.nose_moving_coords_list = []
        self.nose_vector_list = []

    def read_results(self, cap, image, pose_results, hands_results):
        self.cap = cap
        self.frame_width, self.frame_height = int(cap.get(3)), int(cap.get(4))
        self.image = image

        if pose_results.pose_landmarks:
            self.pose_results = pose_results

        if hands_results.multi_hand_landmarks:
            self.hands_results = hands_results

    def get_pose_coords(self, landmark_index):
        return tuple(np.multiply(
            np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                      self.pose_results.pose_landmarks.landmark[landmark_index].y,
                      self.pose_results.pose_landmarks.landmark[landmark_index].z)),
            [self.cap.get(3), self.cap.get(4), self.cap.get(3)]).astype(int))

    def get_exact_pose_coords(self, landmark_index):
        return tuple(np.multiply(
            np.array((self.pose_results.pose_landmarks.landmark[landmark_index].x,
                      self.pose_results.pose_landmarks.landmark[landmark_index].y,
                      self.pose_results.pose_landmarks.landmark[landmark_index].z)),
            [self.cap.get(3), self.cap.get(4), self.cap.get(3)]))

    def get_pose_joint_angle(self, joint):
        co1, co2, co3 = [self.get_exact_pose_coords(joint[i]) for i in range(3)]

        radxy = np.arctan2(co3[1] - co2[1], co3[0] - co2[0]) - np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
        anglexy = np.abs(radxy * 180 / np.pi)
        anglexy = min(anglexy, 360 - anglexy)
        return anglexy

    def show_pose_joint_angles(self, image, joint_list):
        for joint in joint_list:
            joint_angle = self.get_pose_joint_angle(joint)

            cv2.putText(image, str(round(joint_angle, 2)), self.get_pose_coords(joint[1])[:2], cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        return image

    def get_pose_slope_angle(self, index1, index2):
        co1, co2 = self.get_exact_pose_coords(index1), self.get_exact_pose_coords(index2)

        slope_radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
        slope_anglexy = np.abs(slope_radxy * 180 / np.pi)
        # anglexy = min(anglexy, 360 - anglexy)
        return slope_anglexy

    def draw_pose(self):
        self.mp_drawing.draw_landmarks(
            self.image,
            self.pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())

    def detect_hand_raise(self):
        right_shoulder = self.get_exact_pose_coords(12)
        right_elbow = self.get_exact_pose_coords(14)
        right_wrist = self.get_exact_pose_coords(16)
        raised_hand_list = []

        if right_wrist[1] <= right_shoulder[1] and \
                right_wrist[1] <= right_elbow[1] and \
                (120 >= self.get_pose_joint_angle((12, 14, 16)) >= 30 or
                 135 >= self.get_pose_slope_angle(14, 16) >= 45) and 10 <= self.get_pose_slope_angle(14, 16) <= 170:
            print("Right hand raised", right_wrist)
            raised_hand_list.append(("R", right_wrist))

        left_shoulder = self.get_exact_pose_coords(11)
        left_elbow = self.get_exact_pose_coords(13)
        left_wrist = self.get_exact_pose_coords(15)

        if left_wrist[1] <= left_shoulder[1] and \
                left_wrist[1] <= left_elbow[1] and \
                (120 >= self.get_pose_joint_angle((11, 13, 15)) >= 30 or
                 135 >= self.get_pose_slope_angle(13, 15) >= 45) and 10 <= self.get_pose_slope_angle(13, 15) <= 170:
            print("Left hand raised", self.get_pose_coords(15))
            raised_hand_list.append(("L", left_wrist))

        if raised_hand_list:
            cv2.putText(self.image,
                        " ".join([{"R": "Right hand raised", "L": "Left hand raised"}[h[0]] for h in raised_hand_list]),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return raised_hand_list

    def get_distance(self, p1, p2):
        dx, dy, dz = (p2[i] - p1[i] for i in range(3))
        dxy = (dx ** 2 + dy ** 2) ** 0.5

        return dx, dy, dz, dxy

    def is_moving(self, coords_bfat, time_interval, min_distance):
        # global self.last_moving_time
        dx, dy, dz, dxy = self.get_distance(*coords_bfat)
        # print(dxy, min_distance)
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
            # return True
        elif time.time() - self.last_moving_time < time_interval:
            return "o"
        else:
            return False

    def draw_moving_line(self, image, landmark_index, coords_list, is_moving, vector_list, draw_line, color=(255, 0, 0),
                         thickness=3):
        if is_moving:
            if is_moving in "-x-y" and [is_moving] != vector_list[-1:]:
                vector_list.append(is_moving)
            coords_list.append(self.get_pose_coords(landmark_index)[:2])

            if draw_line:
                for start_point, end_point in zip(coords_list, coords_list[1:]):
                    cv2.line(image, start_point, end_point, color, thickness)
            # print(vector_list)
        elif vector_list != []:
            coords_list.clear()
            vector_list.clear()
            print("Head stopped moving")

    def detect_nod(self, time_interval=0.5, min_distance=2, draw_line=True):
        self.nose_coords_bfat = [self.nose_coords_bfat[1], self.get_exact_pose_coords(0)]

        self.draw_moving_line(self.image, 0, self.nose_moving_coords_list,
                              self.is_moving(self.nose_coords_bfat, time_interval=time_interval,
                                             min_distance=min_distance), self.nose_vector_list, draw_line)

        # latest 15 vectors
        lv = self.nose_vector_list[-15:]

        if lv.count("x") > 0 and lv.count("-x") > 0 and lv.count("x") + lv.count("-x") >= 0.7 * len(lv):
            cv2.putText(self.image, "Shaking", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("Shaking detected")
            return "Nodding"
        if lv.count("y") > 0 and lv.count("-y") > 0 and lv.count("y") + lv.count("-y") >= 0.7 * len(lv):
            cv2.putText(self.image, "Nodding", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            print("Nodding detected")
            return "Shaking"

        return "None"

    # hand

    def get_hand_coords(self, hand, landmark_index):
        return tuple(np.multiply(
            np.array(
                (hand.landmark[landmark_index].x, hand.landmark[landmark_index].y, hand.landmark[landmark_index].z)),
            [self.cap.get(3), self.cap.get(4), self.cap.get(3)]).astype(int))

    def get_exact_hand_coords(self, hand, landmark_index):
        return tuple(np.multiply(
            np.array(
                (hand.landmark[landmark_index].x, hand.landmark[landmark_index].y, hand.landmark[landmark_index].z)),
            [self.cap.get(3), self.cap.get(4), self.cap.get(3)]))

    def get_hand_label(self, index, hand, results):
        classification = results.multi_handedness[index]
        label = classification.classification[0].label
        label = ("Right", "Left")[("Left", "Right").index(label)]
        score = classification.classification[0].score
        txt = "{} {}".format(label, round(score, 2))

        # Get coordinates
        coords = self.get_hand_coords(hand, 0)[:2]

        return txt, coords

    def draw_finger_angles(self, image, hand, joint_list):
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
        co1, co2 = self.get_exact_hand_coords(hand, index1), self.get_exact_hand_coords(hand, index2)

        radxy = np.arctan2(co1[1] - co2[1], co1[0] - co2[0])
        # anglexy = radxy * 180 / np.pi
        # anglexy = min(anglexy, 360 - anglexy)
        return radxy

    def get_hand_slope(self, hand, index1, index2):
        co1, co2 = self.get_exact_hand_coords(hand, index1), self.get_exact_hand_coords(hand, index2)
        slope = (co2[1] - co1[1]) / (co2[0] - co1[0])

        return slope

    def draw_cont_line(self, hand, image, start_point, mid_point, length=200, color=(0, 255, 0), thickness=2):
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

        # print(xlen, ylen)
        return co_start, co_mid, slope

    def draw_hand(self):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):
            self.mp_drawing.draw_landmarks(self.image, hand, self.mp_hands.HAND_CONNECTIONS,
                                           self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                           self.mp_drawing_styles.get_default_hand_connections_style())

    def draw_box(self, image, box_name, xywh_tuple, is_pointed):
        x, y, w, h = xywh_tuple
        if is_pointed:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, box_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    def draw_boxes(self, box_list):
        for box in box_list:
            self.draw_box(self.image, *box)

    def draw_hand_label(self):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

            if self.get_hand_label(num, hand, self.hands_results):
                text, coord = self.get_hand_label(num, hand, self.hands_results)
                cv2.putText(self.image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def point_to(self, box_list, finger_list):
        for num, hand in enumerate(self.hands_results.multi_hand_landmarks):

            for boxi in range(len(box_list)):
                box_name, xywh, is_pointed = box_list[boxi]
                bx, by, bw, bh = xywh
                for finger in finger_list:
                    co_start, co_mid, slope = self.draw_cont_line(hand, self.image, *finger, color=(255, 0, 0))
                    # length = finger[2]

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
                        if not ((y_bx < by and y_bxw < by) or (y_bx > by + bh and y_bxw > by + bh)):
                            # set is_pointed to True
                            box_list[boxi][-1] = True

        if [b[0] for b in box_list if b[-1]]:
            cv2.putText(self.image, "Pointed at: " + ",".join([b[0] for b in box_list if b[-1]]),
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            print("Pointed at {}".format([b[0] for b in box_list if b[-1]]))
            return [b[:-1] for b in box_list if b[-1]]
        return "None"
