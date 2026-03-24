import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import glfw
import os
import sys

# ================= 配置区域 =================
DETECTION_MODE = 'color_only'  # 'yolo_color' 或 'color_only'
MODEL_PATH = './ur5e_robotiq2f85/scene.xml'  # 确保路径正确
# ===========================================

# 尝试导入 YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("Warning: ultralytics not found. Falling back to pure color detection.")

# 设置 MuJoCo 渲染后端
os.environ['MUJOCO_GL'] = 'glfw'

class env_cam:
    def __init__(self, model, data, camera_name, width=640, height=480):
        """
        初始化相机类。
        耗时：约 10-50ms/个 (取决于 GPU 驱动和 GLFW 初始化速度)
        位置：在 PickBoxEnv.__init__ -> self._init_cameras() 中被调用
        """
        self.model = model
        self.data = data
        self.name = camera_name
        self.width, self.height = width, height
        self.rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
        # 创建不可见窗口用于离屏渲染
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(self.width, self.height, f"Cam: {camera_name}", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError(f"Could not create GLFW window for {camera_name}")
        glfw.make_context_current(window)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id == -1:
            raise ValueError(f"Camera '{camera_name}' not found in model.")
        self.camera.fixedcamid = camera_id

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)

        self.vopt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        
        self._window = window

    def get_frame(self):
        """获取当前帧的 RGB numpy 数组"""
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.perturb, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, viewport, self.context)
        # MuJoCo reads from bottom-up, flip to top-down
        return np.flipud(self.rgb_buffer).copy()

    def show_img_debug(self, image=None, wait_key=1):
        """显示调试图像到 OpenCV 窗口"""
        if image is None:
            glfw.poll_events()
            cv2.waitKey(1)
            return
        
        # 确保传入的是 BGR 格式给 OpenCV imshow
        # 如果输入是 RGB (来自 get_frame)，需要转换
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 检查是否已经是 BGR (简单启发式：如果 R 通道值普遍很高且 B 很低，可能是 RGB)
            # 这里为了安全，假设外部传入的是 RGB (因为 get_frame 返回 RGB)
            # 但 detect_target_visual 内部已经处理了转换并传回 BGR 用于显示逻辑
            # 为了统一，这里假设传入的是 BGR (由调用者保证) 或者强制转换
            # 修正策略：调用者 (detect_target_visual) 应该传入已经准备好显示的 BGR 图像
            bgr_image = image 
        else:
            bgr_image = image
            
        cv2.imshow(f"Vision: {self.name}", bgr_image)
        key = cv2.waitKey(wait_key)
        glfw.poll_events()
        return key

    def _estimate_world_pos_from_pixel(self, px, py, assumed_z=0.03):
        """简单的针孔相机模型反投影"""
        cam_id = self.camera.fixedcamid
        if cam_id >= self.data.cam_xpos.shape[0]:
            return None
            
        cam_pos = self.data.cam_xpos[cam_id].copy()
        cam_mat_flat = self.data.cam_xmat[cam_id].copy()
        cam_mat = cam_mat_flat.reshape(3, 3)
        
        nx = (px - self.width / 2) / (self.width / 2)
        ny = -(py - self.height / 2) / (self.height / 2) 
        
        fovy = self.model.cam_fovy[cam_id] if cam_id < len(self.model.cam_fovy) else 60.0
        fov = np.radians(fovy)
        aspect = self.width / self.height
        tan_half_fov = np.tan(fov / 2)
        
        ray_cam = np.array([nx * tan_half_fov * aspect, ny * tan_half_fov, 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        
        ray_world = cam_mat @ ray_cam
        
        denom = ray_world[2]
        if abs(denom) < 1e-6: 
            return None 
            
        t = (assumed_z - cam_pos[2]) / denom
        if t < 0: 
            return None 
            
        intersection = cam_pos + t * ray_world
        return intersection

    def close(self):
        if hasattr(self, '_window') and self._window:
            glfw.destroy_window(self._window)

class PickBoxEnv:
    def __init__(self):
        # ================= 运动参数 =================
        self.CARTESIAN_VELOCITY = 0.05
        self.CARTESIAN_ACCELERATION = 0.05
        self.MAX_JOINT_STEP = 0.05
        self.TOLERANCE = 0.002
        self.INTEGRATION_DT = 1.0
        self.IK_DAMPING = 0.0001
        
        self.GRASP_APPROACH_HEIGHT = 0.30
        self.GRASP_HEIGHT = 0.05
        self.LIFT_HEIGHT = 0.30
        self.PLACE_POS = np.array([0.2, 0.0, 0.0])
        # ===========================================

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please check the path.")
            
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # 关节与执行器 ID
        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_ids = []
        for name in joint_names:
            try:
                self.joint_ids.append(self.model.joint(name).id)
            except Exception:
                pass
        
        self.actuator_ids = [i for i in range(self.model.nu) if i < 6]
        
        try:
            self.gripper_id = self.model.actuator("fingers_actuator").id
        except Exception:
            self.gripper_id = self.model.nu - 1 if self.model.nu > 0 else 0

        try:
            self.site_id = self.model.site("attachment_site").id
        except Exception:
            self.site_id = -1 

        # 初始化运动控制变量
        damping = self.IK_DAMPING
        self.jac = np.zeros([6, self.model.nv])
        self.diag = damping * np.eye(6)
        self.error = np.zeros(6)
        
        self.v = self.CARTESIAN_VELOCITY
        self.a = self.CARTESIAN_ACCELERATION
        self.t1, self.t2, self.t3 = 0, 0, 0
        self.d1, self.d2, self.d3 = 0, 0, 0
        self.dx, self.dy, self.dz, self.d = 0, 0, 0, 0
        self.need_plan = True

        # 【关键位置】初始化所有场景相机
        # 耗时：此处发生，每个相机约 10-50ms
        self.cameras = {}
        self._init_cameras()

        # 设置初始状态并前向计算
        self.move2init()
        self.gripper_open()
        mujoco.mj_forward(self.model, self.data)

    def _init_cameras(self):
        """自动发现并初始化所有非内部相机"""
        print("Initializing cameras...")
        start_time = time.time()
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name in ['cam_world', 'cam_tip']: 
                try:
                    cam_obj = env_cam(self.model, self.data, cam_name)
                    self.cameras[cam_name] = cam_obj
                    print(f"  -> Initialized camera: {cam_name}")
                except Exception as e:
                    print(f"  -> Failed to init camera {cam_name}: {e}")
        elapsed = time.time() - start_time
        print(f"Camera initialization completed in {elapsed:.3f} seconds.")

    def reset(self):
        self.move2init()
        self.gripper_open()
        mujoco.mj_forward(self.model, self.data)

    def move2init(self):
        """设置机械臂初始状态并确保其静止"""
        # 定义初始姿态 (弧度)
        init_qpos = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        
        if self.model.njnt >= 6:
            limits = self.model.jnt_range[:6]
            safe_qpos = np.clip(init_qpos, limits[:, 0], limits[:, 1])
            
            # 1. 设置位置
            self.data.qpos[:6] = safe_qpos
            # 2. 清零速度
            self.data.qvel[:6] = 0.0
            # 3. 前向传播
            mujoco.mj_forward(self.model, self.data)
            # 4. 强制将控制目标设置为当前位置 (防止重力下落)
            if self.model.nu >= 6:
                self.data.ctrl[:6] = safe_qpos

        mujoco.mj_forward(self.model, self.data)

    def gripper_open(self): 
        if self.model.nu > 0:
            self.data.ctrl[self.gripper_id] = 0
    
    def gripper_close(self): 
        if self.model.nu > 0:
            self.data.ctrl[self.gripper_id] = 255

    def get_current_pose(self):
        if self.site_id == -1 or self.site_id >= self.model.nsite:
            pos = self.data.xpos[-1].copy()
            rotm_flat = np.eye(9).flatten()
            rotm_33 = np.eye(3)
            return pos, rotm_33, rotm_flat
            
        pos = self.data.site(self.site_id).xpos.copy()
        rotm_flat = self.data.site(self.site_id).xmat.copy() 
        rotm_33 = rotm_flat.reshape(3, 3)
        return pos, rotm_33, rotm_flat

    def line_move(self, start_pos, start_rotm, target_pos, target_euler, start_time, cur_time):
        if self.need_plan:
            self.line_plan(start_pos, target_pos)
            self.start_euler = self.rotm2rpy(start_rotm)
            self.target_euler_full = np.array(target_euler)
        
        t = cur_time - start_time
        if self.t3 == 0: 
            return True, target_pos, target_euler

        if t < self.t1:
            deltaD = 0.5 * self.a * t**2
        elif t < self.t2:
            deltaD = self.d1 + self.v * (t - self.t1)
        elif t <= self.t3:
            deltaT = t - self.t2
            deltaD = self.d2 + self.v * deltaT - 0.5 * self.a * deltaT**2
        else:
            return True, target_pos, target_euler

        ratio = deltaD / self.d if self.d > 0 else 1.0
        curr_p = start_pos + np.array([self.dx, self.dy, self.dz]) * ratio
        curr_e = self.start_euler + (self.target_euler_full - self.start_euler) * ratio
        
        self.move2pose(curr_p, curr_e, flag=False)
        return False, curr_p, curr_e

    def line_plan(self, start_pos, end_pos):
        v, a = self.v, self.a
        dx, dy, dz = end_pos - start_pos
        d = np.linalg.norm([dx, dy, dz])
        self.dx, self.dy, self.dz, self.d = dx, dy, dz, d
        
        if d < 1e-6:
            self.t1 = self.t2 = self.t3 = 0
            return

        if d > v**2 / a:
            t1 = v / a
            d1 = 0.5 * a * t1**2
            d_const = d - 2*d1
            if d_const < 0: 
                 t1 = np.sqrt(d/a)
                 self.t1 = self.t2 = t1
                 self.t3 = 2*t1
                 self.d1 = self.d2 = d/2
                 self.d3 = d
            else:
                self.t1 = t1
                self.t2 = t1 + d_const/v
                self.t3 = self.t2 + t1
                self.d1 = d1
                self.d2 = d1 + d_const
                self.d3 = d
        else:
            t1 = np.sqrt(d/a)
            self.t1 = self.t2 = t1
            self.t3 = 2*t1
            self.d1 = self.d2 = d/2
            self.d3 = d
        self.need_plan = False

    def move2pose(self, target_pos, target_euler, flag=True):
        cur_pos, cur_rotm_33, cur_rotm_flat = self.get_current_pose()
        self.error[:3] = target_pos - cur_pos

        target_quat = np.zeros(4)
        if flag:
            mujoco.mju_euler2Quat(target_quat, target_euler, 'XYZ')
        
        cur_quat = np.zeros(4)
        mujoco.mju_mat2Quat(cur_quat, cur_rotm_flat)
        
        cur_quat_conj = np.zeros(4)
        mujoco.mju_negQuat(cur_quat_conj, cur_quat)
        
        error_quat = np.zeros(4)
        mujoco.mju_mulQuat(error_quat, target_quat, cur_quat_conj)
        mujoco.mju_quat2Vel(self.error[3:], error_quat, 1.0)

        if np.linalg.norm(self.error[:3]) < self.TOLERANCE and np.linalg.norm(self.error[3:]) < 0.01:
            return True

        if self.site_id != -1:
            mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
        else:
            return True 

        A = self.jac @ self.jac.T + self.diag
        try:
            dq = self.jac.T @ np.linalg.solve(A, self.error)
        except np.linalg.LinAlgError:
            return True
            
        dq = np.clip(dq, -self.MAX_JOINT_STEP, self.MAX_JOINT_STEP)

        q = self.data.qpos.copy()
        mujoco.mj_integratePos(self.model, q, dq, self.INTEGRATION_DT)
        
        if self.model.njnt >= 6:
            limits = self.model.jnt_range[:6]
            np.clip(q[:6], limits[:, 0], limits[:, 1], out=q[:6])
            self.data.ctrl[self.actuator_ids] = q[:6]
        return False

    def rotm2rpy(self, R):
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        if np.isclose(abs(pitch), np.pi / 2):
            yaw = 0
            roll = np.arctan2(R[0, 1], R[1, 1])
        else:
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
        return np.array([roll, pitch, yaw])

# ================= 视觉检测函数 =================

def get_blue_mask(hsv_img):
    # 蓝色在 HSV 中的范围 (OpenCV H: 0-180)
    # 蓝色通常在 100-130 之间
    lower_blue = np.array([100, 150, 150], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_target_visual(env, model_yolo=None):
    """
    检测目标并返回结果。
    【关键修复】正确处理 RGB -> BGR -> HSV 的颜色空间转换。
    """
    if not env.cameras:
        return False, None, None, {}

    best_result = None
    max_area_global = 0
    debug_images = {}

    primary_cam_name = 'cam_world' if 'cam_world' in env.cameras else list(env.cameras.keys())[0]
    
    for name, cam_obj in env.cameras.items():
        # 1. 获取帧 (RGB 格式)
        frame_rgb = cam_obj.get_frame()
        
        # 2. 【修复】必须先转为 BGR，再转为 HSV
        # MuJoCo 输出 RGB, OpenCV 期望 BGR 进行大部分操作，特别是 cvtColor(BGR2HSV)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # 3. 转为 HSV 进行颜色分割
        hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        h, w, _ = frame_bgr.shape

        best_cnt = None
        max_area = 0
        cx, cy = 0, 0

        if DETECTION_MODE == 'yolo_color' and model_yolo is not None and HAS_YOLO:
            results = model_yolo(frame_bgr, verbose=False, conf=0.4)
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    roi = hsv_img[y1:y2, x1:x2]
                    if roi.size == 0: continue
                    mean_hue = np.mean(roi[:, :, 0])
                    if 100 <= mean_hue <= 130: # 蓝色范围
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            cx, cy = center_x, center_y
                            best_cnt = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        
        elif DETECTION_MODE == 'color_only':
            mask = get_blue_mask(hsv_img)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:
                    if area > max_area:
                        max_area = area
                        best_cnt = cnt
                        x, y, wr, hr = cv2.boundingRect(cnt)
                        cx, cy = x + wr // 2, y + hr // 2

        # 准备调试图像 (使用 BGR 格式以便 OpenCV 显示)
        debug_img = frame_bgr.copy()
        label_status = "Searching..."
        
        if max_area > 0 and best_cnt is not None:
            if DETECTION_MODE == 'color_only':
                cv2.drawContours(debug_img, [best_cnt], -1, (0, 255, 0), 2)
            cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), -1)
            label_status = f"Detected (Area: {max_area})"
            
            if name == primary_cam_name:
                assumed_z_height = 0.04
                world_pos = cam_obj._estimate_world_pos_from_pixel(cx, cy, assumed_z=assumed_z_height)
                if world_pos is not None:
                    label_pos = f"Pos: [{world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}]"
                    cv2.putText(debug_img, label_pos, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if max_area > max_area_global:
                        max_area_global = max_area
                        best_result = (True, world_pos, 0.0)
                else:
                    best_result = (False, None, None)
        
        cv2.putText(debug_img, f"{name}: {label_status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        debug_images[name] = debug_img

    if best_result is None:
        best_result = (False, None, None)

    return best_result[0], best_result[1], best_result[2], debug_images

# ================= 主程序 =================

if __name__ == "__main__":
    env = None
    yolo_model = None

    try:
        print("Loading MuJoCo Model...")
        env = PickBoxEnv()
        
        if DETECTION_MODE == 'yolo_color' and HAS_YOLO:
            print("Loading YOLOv11 model...")
            try:
                yolo_model = YOLO('yolo11n.pt') 
            except Exception as e:
                print(f"Failed to load YOLO: {e}. Using color-only logic.")
                yolo_model = None

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            env.reset()
            
            print("Warming up visualization and cameras...")
            # 预热循环
            for _ in range(100):
                # 在预热期间也锁定控制，防止微小漂移
                if env.model.nu >= 6:
                    env.data.ctrl[:6] = env.data.qpos[:6]
                
                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                for cam in env.cameras.values():
                    _ = cam.get_frame()
                glfw.poll_events()

            print("=== STARTING DETECTION PHASE (Robot Stationary) ===")
            print("Press Ctrl+C in terminal to abort.")
            
            target_pos = None
            target_euler_z = 0.0
            
            # --- 阶段 1: 静止检测循环 ---
            detected = False
            while not detected and viewer.is_running():
                step_start = time.time()
                
                found, pos, euler_z, debug_imgs = detect_target_visual(env, yolo_model)
                
                for name, img in debug_imgs.items():
                    if name in env.cameras:
                        env.cameras[name].show_img_debug(img, wait_key=1)
                
                if found:
                    print(f"\n🎯 TARGET LOCKED!")
                    print(f"   Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                    target_pos = pos
                    target_euler_z = euler_z
                    detected = True

                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                
                elapsed = time.time() - step_start
                if elapsed < env.dt:
                    time.sleep(env.dt - elapsed)

            if not detected:
                print("Detection loop ended without finding target.")
                sys.exit(0)

            # --- 阶段 2: 执行抓取任务 ---
            print("=== STARTING MANIPULATION SEQUENCE ===")
            
            step = 0 
            step_res = True 
            start_pos = None
            start_rotm = None
            start_time = 0
            target = np.zeros(6)

            while viewer.is_running():
                step_start = time.time()
                
                if step == 0 and step_res:
                    print("  -> Step 0: Moving to Approach Height...")
                    step = 1; step_res = False; env.need_plan = True
                    current_pose = env.get_current_pose()
                    start_pos, start_rotm = current_pose[0], current_pose[1]
                    start_time = env.data.time
                    
                    target[:3] = target_pos + np.array([0.0, 0.0, env.GRASP_APPROACH_HEIGHT]) 
                    target[3:] = [np.pi, 0.0, np.pi / 2]

                elif step == 1 and step_res:
                    print("  -> Step 1: Moving Down to Grasp...")
                    step = 2; step_res = False; env.need_plan = True
                    current_pose = env.get_current_pose()
                    start_pos, start_rotm = current_pose[0], current_pose[1]
                    start_time = env.data.time
                    
                    target[:3] = target_pos + np.array([0.0, 0.0, env.GRASP_HEIGHT])
                    target[3:] = [np.pi, 0.0, np.pi / 2 + target_euler_z]

                elif step == 2 and step_res:
                    print("  -> Step 2: Closing Gripper & Lifting...")
                    env.gripper_close()
                    for _ in range(30):
                        mujoco.mj_step(env.model, env.data)
                        viewer.sync()
                    
                    step = 3; step_res = False; env.need_plan = True
                    current_pose = env.get_current_pose()
                    start_pos, start_rotm = current_pose[0], current_pose[1]
                    start_time = env.data.time
                    
                    target[:3] = target_pos + np.array([0.0, 0.0, env.LIFT_HEIGHT])
                    target[3:] = [np.pi, 0.0, np.pi / 2]

                elif step == 3 and step_res:
                    print("  -> Step 3: Moving to Place Location...")
                    step = 4; step_res = False; env.need_plan = True
                    current_pose = env.get_current_pose()
                    start_pos, start_rotm = current_pose[0], current_pose[1]
                    start_time = env.data.time
                    
                    place_target = env.PLACE_POS + np.array([0.0, 0.0, env.LIFT_HEIGHT])
                    target[:3] = place_target
                    target[3:] = [np.pi, 0.0, np.pi / 2]

                elif step == 4 and step_res:
                    print("  -> Step 4: Moving Down to Place...")
                    step = 5; step_res = False; env.need_plan = True
                    current_pose = env.get_current_pose()
                    start_pos, start_rotm = current_pose[0], current_pose[1]
                    start_time = env.data.time
                    
                    target[:3] = env.PLACE_POS + np.array([0.0, 0.0, env.GRASP_HEIGHT])
                    target[3:] = [np.pi, 0.0, np.pi / 2]

                elif step == 5 and step_res:
                    print("  -> Step 5: Opening Gripper & Retract...")
                    env.gripper_open()
                    for _ in range(30):
                        mujoco.mj_step(env.model, env.data)
                        viewer.sync()
                    
                    step = 6
                    print("✅ Task Completed Successfully!")
                    break 

                if step < 6:
                    current_time = env.data.time
                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, 
                        target[:3], target[3:], 
                        start_time, current_time
                    )
                    if done_move:
                        step_res = True

                _, _, _, debug_imgs = detect_target_visual(env, yolo_model)
                for name, img in debug_imgs.items():
                    if name in env.cameras:
                        env.cameras[name].show_img_debug(img, wait_key=1)

                mujoco.mj_step(env.model, env.data)
                viewer.sync()

                time_until_next_step = env.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Shutting down...")
        if env:
            env.gripper_open()
            env.move2init()
            for cam in env.cameras.values():
                cam.close()
        glfw.terminate()
        cv2.destroyAllWindows()
        print("System Shutdown.")
