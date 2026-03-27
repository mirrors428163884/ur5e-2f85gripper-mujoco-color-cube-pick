import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import glfw
import os
import yaml

os.environ['MUJOCO_GL'] = 'glfw'


class env_cam():
    def __init__(self, model, data, camera_name, width=640, height=480):
        self.model = model
        self.data = data
        self.name = camera_name

        self.width, self.height = width, height
        self.rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.width, self.height, camera_name, None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        glfw.make_context_current(window)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        self.camera.fixedcamid = camera_id

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.vopt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()

    def show_img(self, show=True):
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.perturb, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, viewport, self.context)
        rgb_image = np.flipud(self.rgb_buffer)

        if show:
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.name, bgr_image)
            cv2.waitKey(1)

        glfw.poll_events()


class PickBoxEnv():
    def __init__(self, config_path='config.yaml'):
        # 加载配置文件
        self.load_config(config_path)
        
        self.model = mujoco.MjModel.from_xml_path('./ur5e_robotiq2f85/scene.xml')
        self.data = mujoco.MjData(self.model)

        joint_names = ["shoulder_pan_joint",
                      "shoulder_lift_joint",
                      "elbow_joint",
                      "wrist_1_joint",
                      "wrist_2_joint",
                      "wrist_3_joint"]
        self.joint_ids = [self.model.joint(name).id for name in joint_names]
        self.actuator_ids = [self.model.actuator(i).id for i in range(self.model.nu)]
        self.actuator_ids = self.actuator_ids[:6]
        self.gripper_id = self.model.actuator("fingers_actuator").id
        self.site_id = self.model.site("attachment_site").id
        self.box1_id = self.model.body("box1").id
        self.box2_id = self.model.body("box2").id
        self.box3_id = self.model.body("box3").id

        # 从配置文件加载参数
        motion_config = self.config['motion']
        self.MAX_JOINT_STEP = motion_config['max_joint_step']
        self.TOLERANCE = motion_config['tolerance']
        self.integration_dt = motion_config['integration_dt']
        damping = motion_config['ik_damping']
        self.dt = self.config['simulation']['dt']

        self.jac = np.zeros([6, self.model.nv])
        self.diag = damping * np.eye(6)
        self.error = np.zeros(6)
        self.error_pos = self.error[:3]
        self.error_ori = self.error[3:]

        self.cur_pos = np.zeros(3)
        self.cur_rotm = np.zeros([3, 3])
        self.cur_quat = np.zeros(4)
        self.cur_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        self.move2init()

        # 从配置文件加载笛卡尔运动参数
        self.v = motion_config['cartesian_velocity']
        self.a = motion_config['cartesian_acceleration']
        self.t1, self.t2, self.t3 = 0, 0, 0
        self.d1, self.d2, self.d3 = 0, 0, 0
        self.dx, self.dy, self.dz, self.d = 0, 0, 0, 0

        self.seed = None
        self.set_seed()

        self.cam_world = env_cam(self.model, self.data, 'cam_world')
        self.cam_tip = env_cam(self.model, self.data, 'cam_tip')

    def load_config(self, config_path):
        """加载YAML配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 未找到，使用默认配置")
            self.config = {
                'motion': {
                    'cartesian_velocity': 0.05,
                    'cartesian_acceleration': 0.05,
                    'max_joint_step': 0.05,
                    'tolerance': 0.002,
                    'integration_dt': 1.0,
                    'ik_damping': 0.0001
                },
                'simulation': {
                    'dt': 0.002,
                    'sleep_control': True
                },
                'boxes': {
                    'red': 0,
                    'green': 1,
                    'blue': 2,
                    'random_range': {
                        'x': [-0.7, -0.4],
                        'y': [-0.2, 0.2],
                        'z': 0.03,
                        'rotation_z': [-0.785, 0.785]
                    }
                },
                'grasping': {
                    'approach_height': 0.3,
                    'grasp_height': 0.17,
                    'lift_height': 0.3,
                    'place_height': 0.20
                }
            }

    def reset(self, pick_color=None, place_color=None):
        """重置环境，可指定抓取和放置的颜色"""
        self.move2init()
        self.gripper_open()
        self.set_box_pos_random()
        
        if pick_color is not None and place_color is not None:
            # 验证颜色有效性
            valid_colors = ['red', 'green', 'blue']
            if pick_color not in valid_colors or place_color not in valid_colors:
                raise ValueError(f"颜色必须是 {valid_colors} 中的一个")
            if pick_color == place_color:
                raise ValueError("抓取和放置的颜色不能相同")
            
            color_map = {'red': 0, 'green': 1, 'blue': 2}
            self.pick_box = color_map[pick_color]
            self.place_box = color_map[place_color]
        else:
            self.choose_box()

    def set_seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def line_plan(self, start_pos, start_rotm, end_pos, end_euler):
        v = self.v
        a = self.a

        self.dx = end_pos[0] - start_pos[0]
        self.dy = end_pos[1] - start_pos[1]
        self.dz = end_pos[2] - start_pos[2]
        self.d = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        d = self.d
        if d > v**2 / a:
            t1 = v / a
            d1 = 0.5 * a * t1**2
            d2 = d - d1
            t2 = t1 + (d2 - d1) / v
            d3 = d
            t3 = t2 + t1
        else:
            d1 = d / 2.0
            t1 = np.sqrt(d / a)
            t2 = t1
            d2 = d1
            t3 = 2.0 * t1
            d3 = d
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.need_plan = False

    def line_move(self, start_pos, start_rotm, end_pos, end_euler, start_time, cur_time):
        if self.need_plan:
            self.line_plan(start_pos, start_rotm, end_pos, end_euler)
        t = cur_time - start_time
        if t < self.t1:
            deltaD = 0.5 * self.a * t**2
        elif t < self.t2:
            deltaD = self.d1 + self.v * (t - self.t1)
        elif t <= self.t3:
            deltaT = t - self.t2
            deltaD = self.d2 + self.a * deltaT * (self.t1 - 0.5 * deltaT)
        else:
            return True, end_pos, end_euler
        x = start_pos[0] + self.dx * deltaD / self.d
        y = start_pos[1] + self.dy * deltaD / self.d
        z = start_pos[2] + self.dz * deltaD / self.d
        self.move2pose(np.array([x, y, z]), end_euler, False)
        return False, np.array([x, y, z]), end_euler

    def get_current_pose(self):
        pos = self.data.site(self.site_id).xpos.copy()
        rotm = self.data.site(self.site_id).xmat.copy()
        return pos, rotm

    def move2pose(self, target_pos, target_rotm, flag):
        self.cur_pos, self.cur_rotm = self.get_current_pose()
        self.error_pos[:] = target_pos - self.cur_pos

        target_quat = np.zeros((4))
        if flag:
            mujoco.mju_mat2Quat(target_quat, target_rotm)
        else:
            if len(target_rotm) != 3:
                print(11)
            mujoco.mju_euler2Quat(target_quat, target_rotm, 'XYZ')

        mujoco.mju_mat2Quat(self.cur_quat, self.cur_rotm)
        mujoco.mju_negQuat(self.cur_quat_conj, self.cur_quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.cur_quat_conj)
        mujoco.mju_quat2Vel(self.error_ori, self.error_quat, 1.0)

        if np.linalg.norm(self.error[:3]) < self.TOLERANCE:
            return True

        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        dq = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + self.diag, self.error)
        dq = np.clip(dq, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)

        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.integration_dt)
        np.clip(q[:6], *self.model.jnt_range[:6].T, out=q[:6])
        self.data.ctrl[self.actuator_ids] = q[self.joint_ids]
        return False

    def gripper_open(self):
        self.data.ctrl[self.gripper_id] = 0

    def gripper_close(self):
        self.data.ctrl[self.gripper_id] = 255

    def move2init(self):
        self.data.qpos[:6] = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        mujoco.mj_forward(self.model, self.data)

    def set_box_pos_random(self):
        box_config = self.config['boxes']['random_range']
        x_range = box_config['x']
        y_range = box_config['y']
        z_pos = box_config['z']
        rot_range = box_config['rotation_z']

        self.data.qpos[self.model.body_jntadr[self.box1_id]:self.model.body_jntadr[self.box1_id] + 3] = [
            self.rng.uniform(x_range[0], x_range[1]), 
            self.rng.uniform(y_range[0], y_range[1]), 
            z_pos
        ]
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 7:self.model.body_jntadr[self.box1_id] + 10] = [
            self.rng.uniform(x_range[0], x_range[1]), 
            self.rng.uniform(y_range[0], y_range[1]), 
            z_pos
        ]
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 14:self.model.body_jntadr[self.box1_id] + 17] = [
            self.rng.uniform(x_range[0], x_range[1]), 
            self.rng.uniform(y_range[0], y_range[1]), 
            z_pos
        ]
        
        quat = np.zeros(4)
        z1 = self.rng.uniform(rot_range[0], rot_range[1])
        z2 = self.rng.uniform(rot_range[0], rot_range[1])
        z3 = self.rng.uniform(rot_range[0], rot_range[1])

        mujoco.mju_euler2Quat(quat, [.0, .0, z1], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 3:self.model.body_jntadr[self.box1_id] + 7] = quat

        mujoco.mju_euler2Quat(quat, [.0, .0, z2], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 10:self.model.body_jntadr[self.box1_id] + 14] = quat

        mujoco.mju_euler2Quat(quat, [.0, .0, z3], 'xyz')
        self.data.qpos[self.model.body_jntadr[self.box1_id] + 17:self.model.body_jntadr[self.box1_id] + 21] = quat

        mujoco.mj_forward(self.model, self.data)
        return np.array([z1, z2, z3])

    def get_box_pos(self, color):
        box_id = -1
        if color == 0:  # red
            box_id = self.box1_id
        elif color == 1:  # green
            box_id = self.box2_id
        else:  # blue
            box_id = self.box3_id
        pos = self.data.body(box_id).xpos.copy()
        rotm = self.data.body(box_id).xmat.copy()
        euler = self.rotm2rpy(rotm.reshape(3, 3))
        eulerz = euler[2]
        return pos, eulerz

    def choose_box(self):
        arr = np.array([0, 1, 2])
        self.pick_box, self.place_box = self.rng.choice(arr, size=2, replace=False)

    def rotm2rpy(self, R):
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if np.isclose(abs(pitch), np.pi / 2):
            yaw = 0
            roll = np.arctan2(R[0, 1], R[1, 1])
        else:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])
