from smplx.joint_names import JOINT_NAMES

class Fit3DOrder26P:
    pelvis = 0
    left_hip = 1
    right_hip = 2
    spine1 = 3
    left_knee = 4
    right_knee = 5
    spine2 = 6
    left_ankle = 7
    right_ankle = 8
    spine3 = 9
    left_foot = 10
    right_foot = 11
    neck = 12
    left_collar = 13
    right_collar = 14
    head = 15
    left_shoulder = 16
    right_shoulder = 17
    left_elbow = 18
    right_elbow = 19
    left_wrist = 20
    right_wrist = 21
    left_thumb3 = 22
    right_thumb3 = 23
    left_pinky3 = 24  # 33
    right_pinky3 = 25  # 48

    _num_joints = 26

    _fingers_smplx = [39, 54, 33, 48]
    _fingers_pose = [14, 8]

    _left_fingers = [22, 24]
    _right_fingers = [23, 25]

    _body_smplx = list(range(0, 22))
    global_orient = pelvis

    _point_names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_thumb3",
        "right_thumb3",
        "left_pinky3",
        "right_pinky3",
    ]

    _indices = [
        pelvis,
        left_hip,
        right_hip,
        spine1,
        left_knee,
        right_knee,
        spine2,
        left_ankle,
        right_ankle,
        spine3,
        left_foot,
        right_foot,
        neck,
        left_collar,
        right_collar,
        head,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_thumb3,
        right_thumb3,
        left_pinky3,
        right_pinky3,
    ]

    _flip_lr_indices = [
        pelvis,
        right_hip,
        left_hip,
        spine1,
        right_knee,
        left_knee,
        spine2,
        right_ankle,
        left_ankle,
        spine3,
        right_foot,
        left_foot,
        neck,
        right_collar,
        left_collar,
        head,
        right_shoulder,
        left_shoulder,
        right_elbow,
        left_elbow,
        right_wrist,
        left_wrist,
        right_thumb3,
        left_thumb3,
        right_pinky3,
        left_pinky3,
    ]

    _flipped_indices = [
        left_hip,
        right_hip,
        left_knee,
        right_knee,
        left_ankle,
        right_ankle,
        left_foot,
        right_foot,
        left_collar,
        right_collar,
        left_shoulder,
        right_shoulder,
        left_elbow,
        right_elbow,
        left_wrist,
        right_wrist,
        left_thumb3,
        right_thumb3,
        left_pinky3,
        right_pinky3,
    ]

    _kinematic_tree = [
        [pelvis, left_hip],
        [pelvis, right_hip],
        [pelvis, spine1],
        [left_hip, left_knee],
        [right_hip, right_knee],
        [spine1, spine2],
        [left_knee, left_ankle],
        [right_knee, right_ankle],
        [spine2, spine3],
        [left_ankle, left_foot],
        [right_ankle, right_foot],
        [spine3, neck],
        [spine3, left_collar],
        [spine3, right_collar],
        [neck, head],
        [left_collar, left_shoulder],
        [right_collar, right_shoulder],
        [left_shoulder, left_elbow],
        [right_shoulder, right_elbow],
        [left_elbow, left_wrist],
        [right_elbow, right_wrist],
        [left_wrist, left_thumb3],
        [right_wrist, right_thumb3],
        [left_wrist, left_pinky3],
        [right_wrist, right_pinky3],
    ]

    _bones = [
        [(left_elbow, left_wrist), (right_elbow, right_wrist)],
        [(left_shoulder, left_elbow), (right_shoulder, right_elbow)],
        [(left_knee, left_ankle), (right_knee, right_ankle)],
        [(left_hip, left_knee), (right_hip, right_knee)],
    ]

    _hands = [left_thumb3, right_thumb3, right_pinky3, left_pinky3]

    @classmethod
    def bones(cls):
        return cls._bones

    @classmethod
    def names(cls):
        return cls._point_names

    @classmethod
    def indices(cls):
        return cls._indices

    @classmethod
    def flip_lr_indices(cls):
        return cls._flip_lr_indices

    @classmethod
    def body_pose_smplx(cls):
        return cls._body_smplx[1:]

    @classmethod
    def smplx_joints(cls):
        return cls._body_smplx + cls._fingers_smplx

    @classmethod
    def hands(cls):
        return cls._hands


class SMPLX37Order:
    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_index",
        "left_thumb",
        "right_index",
        "right_thumb",
        "left_big_toe",
        "left_small_toe",
        "left_heel",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
        "right_eye",
        "left_eye",
        "right_ear",
        "left_ear",
        "nose",
    ]

    num_joints = len(names)

    @classmethod
    def indices(cls):
        return cls.from_SMPLX_order()

    @classmethod
    def index(cls, name):
        return cls.names.index(name)

    @classmethod
    def from_SMPLX_order(cls):
        return [JOINT_NAMES.index(name) for name in cls.names]

    @classmethod
    def select_joints_order(cls, order):
        return [cls.names.index(joint) for joint in order.point_names]