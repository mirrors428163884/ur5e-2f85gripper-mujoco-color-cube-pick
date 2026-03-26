import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import glfw
import os
import sys
import threading
import queue
import copy

# ================= 配置区域 =================
DETECTION_MODE = 'color_only'
MODEL_PATH = './ur5e_robotiq2f85/scene.xml'

# 【新增】手眼标定补偿参数
VISION_OFFSET = np.array([0.0025, -0.0015, 0.0]) 

# 【新增】放置释放高度阈值
RELEASE_HEIGHT_THRESHOLD = 0.06 

# 【新增】自由落体仿真持续时间 (秒)
SIMULATION_HOLD_TIME = 2.0 

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

os.environ['MUJOCO_GL'] = 'glfw'

# ================= 线程安全通信类 =================
class FrameBuffer:
    def __init__(self, maxsize=2):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_result = {
            'found': False,
            'pos': None,
            'euler_z': 0.0,
            'debug_img': None,
            'pixel_coords': None,
            'timestamp': 0.0
        }
        self.lock = threading.Lock()
        self.stop_flag = False

    def put_frame(self, cam_name, image_rgb):
        if self.stop_flag:
            return
        try:
            # 非阻塞放入，满则丢弃旧帧，保证实时性
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put((cam_name, image_rgb.copy()), block=False)
        except queue.Full:
            pass

    def set_result(self, found, pos, euler_z, debug_img, pixel_coords=None):
        with self.lock:
            # 深拷贝图像防止外部修改
            img_copy = None
            if debug_img is not None:
                img_copy = debug_img.copy()
            
            self.latest_result['found'] = found
            self.latest_result['pos'] = pos
            self.latest_result['euler_z'] = euler_z
            self.latest_result['debug_img'] = img_copy
            self.latest_result['pixel_coords'] = pixel_coords
            self.latest_result['timestamp'] = time.time()

    def get_result(self):
        with self.lock:
            # 返回副本
            res = self.latest_result.copy()
            if res['debug_img'] is not None:
                res['debug_img'] = res['debug_img'].copy()
            return res

    def stop(self):
        self.stop_flag = True

frame_buffer = FrameBuffer()

class env_cam:
    def __init__(self, model, data, camera_name, width=640, height=480):
        self.model = model
        self.data = data
        self.name = camera_name
        self.width, self.height = width, height
        self.rgb_buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        
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
        self.ready = False

    def get_frame(self):
        if self.model is None or self.data is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, self.perturb, self.camera, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        mujoco.mjr_render(viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.rgb_buffer, None, viewport, self.context)
        
        frame = np.flipud(self.rgb_buffer).copy()
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
        return frame

    def show_img_debug(self, image=None, wait_key=1):
        if image is None:
            glfw.poll_events()
            cv2.waitKey(1)
            return
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image
            
        if not bgr_image.flags['C_CONTIGUOUS']:
            bgr_image = np.ascontiguousarray(bgr_image)

        cv2.imshow(f"Vision: {self.name}", bgr_image)
        key = cv2.waitKey(wait_key)
        glfw.poll_events()
        return key

    def _estimate_world_pos_from_pixel(self, px, py, assumed_z=0.03, debug=False):
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
        
        ray_cam = np.array([nx * tan_half_fov * aspect, ny * tan_half_fov, -1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        
        ray_world = cam_mat @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)

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
            try:
                glfw.destroy_window(self._window)
            except:
                pass

class PickBoxEnv:
    def __init__(self):
        self.CARTESIAN_VELOCITY_SAFE = 0.20
        self.CARTESIAN_ACCELERATION_SAFE = 0.40
        self.MAX_JOINT_STEP_SAFE = 0.05
        self.TOLERANCE_SAFE = 0.005
        
        self.CARTESIAN_VELOCITY_FAST = 2.50
        self.CARTESIAN_ACCELERATION_FAST = 5.00
        self.MAX_JOINT_STEP_FAST = 0.50
        self.TOLERANCE_FAST = 0.02
        
        self.CARTESIAN_VELOCITY = self.CARTESIAN_VELOCITY_SAFE
        self.CARTESIAN_ACCELERATION = self.CARTESIAN_ACCELERATION_SAFE
        self.MAX_JOINT_STEP = self.MAX_JOINT_STEP_SAFE
        self.TOLERANCE = self.TOLERANCE_SAFE
        
        self.high_speed_mode = False
        self.INTEGRATION_DT = 1.0
        self.IK_DAMPING = 0.0001
        
        self.GRASP_APPROACH_HEIGHT = 0.30
        self.GRASP_HEIGHT = 0.05
        self.LIFT_HEIGHT = 0.30
        
        self.BASE_PLACE_POS = np.array([0.2, 0.0, 0.0])

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")
            
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

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

        self.cameras = {}
        self._init_cameras()

        self.move2init()
        self.gripper_open()
        mujoco.mj_forward(self.model, self.data)

        self._print_object_positions()

    def compensate_offset(self, raw_pos):
        if raw_pos is None:
            return None
        compensated_pos = raw_pos + VISION_OFFSET
        print(f"   [Compensation] Raw: [{raw_pos[0]:.4f}, {raw_pos[1]:.4f}, {raw_pos[2]:.4f}] -> Compensated: [{compensated_pos[0]:.4f}, {compensated_pos[1]:.4f}, {compensated_pos[2]:.4f}]")
        return compensated_pos

    def _print_object_positions(self):
        print("\n=== SCENE OBJECT INITIALIZATION ===")
        found_objects = False
        for body_id in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and 'box' in body_name.lower():
                pos = self.data.xpos[body_id].copy()
                print(f"  -> Object: '{body_name}' | Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
                found_objects = True
        if not found_objects:
            print("  -> No objects found.")
        print(f"  -> Vision Offset: {VISION_OFFSET}")
        print("=====================================\n")

    def enable_high_speed_mode(self):
        if self.high_speed_mode:
            return
        print("\n⚡ SWITCHING TO HIGH-SPEED MODE! ⚡")
        self.high_speed_mode = True
        self.CARTESIAN_VELOCITY = self.CARTESIAN_VELOCITY_FAST
        self.CARTESIAN_ACCELERATION = self.CARTESIAN_ACCELERATION_FAST
        self.MAX_JOINT_STEP = self.MAX_JOINT_STEP_FAST
        self.TOLERANCE = self.TOLERANCE_FAST
        self.v = self.CARTESIAN_VELOCITY
        self.a = self.CARTESIAN_ACCELERATION
        self.need_plan = True

    def _init_cameras(self):
        print("Initializing cameras...")
        for i in range(self.model.ncam):
            cam_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if cam_name in ['cam_world', 'cam_tip']: 
                try:
                    cam_obj = env_cam(self.model, self.data, cam_name)
                    self.cameras[cam_name] = cam_obj
                    print(f"  -> Initialized camera: {cam_name}")
                except Exception as e:
                    print(f"  -> Failed to init camera {cam_name}: {e}")

    def reset(self):
        self.move2init()
        self.gripper_open()
        mujoco.mj_forward(self.model, self.data)

    def move2init(self):
        init_qpos = [0, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        if self.model.njnt >= 6:
            limits = self.model.jnt_range[:6]
            if self.high_speed_mode:
                safe_qpos = self.data.qpos[:6].copy()
                self.data.qvel[:6] *= 0.5 
            else:
                safe_qpos = np.clip(init_qpos, limits[:, 0], limits[:, 1])
                self.data.qpos[:6] = safe_qpos
                self.data.qvel[:6] = 0.0
            mujoco.mj_forward(self.model, self.data)
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

        if t < 0: t = 0 

        if t < self.t1:
            deltaD = 0.5 * self.a * t**2
        elif t < self.t2:
            deltaD = self.d1 + self.v * (t - self.t1)
        elif t <= self.t3:
            deltaT = t - self.t2
            deltaD = self.d2 + self.v * deltaT - 0.5 * self.a * deltaT**2
        else:
            return True, target_pos, target_euler

        ratio = deltaD / self.d if self.d > 1e-6 else 1.0
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

        if np.linalg.norm(self.error[:3]) < self.TOLERANCE and np.linalg.norm(self.error[3:]) < (self.TOLERANCE * 5):
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
    lower_blue = np.array([90, 100, 100], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=3)
    return mask

def process_visual_feedback(env, frame_rgb, cam_name):
    if frame_rgb is None or frame_rgb.size == 0:
        return None, False, None

    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    
    best_cnt = None
    max_area = 0
    cx, cy = 0, 0
    found_local = False
    
    if DETECTION_MODE == 'color_only':
        mask = get_blue_mask(hsv_img)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300: 
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
                    x, y, wr, hr = cv2.boundingRect(cnt)
                    cx, cy = x + wr // 2, y + hr // 2
    
    debug_img = frame_bgr.copy()
    label_status = "Searching..."
    
    if max_area > 0 and best_cnt is not None:
        cv2.drawContours(debug_img, [best_cnt], -1, (0, 255, 0), 2)
        cv2.circle(debug_img, (cx, cy), 8, (0, 0, 255), -1)
        label_status = f"Tracking! Area: {max_area}"
        found_local = True
    
    cv2.putText(debug_img, f"{cam_name}: {label_status}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    frame_buffer.set_result(
        found=found_local,
        pos=None,
        euler_z=0.0,
        debug_img=debug_img,
        pixel_coords=(cx, cy) if found_local else None
    )
    
    return debug_img, found_local, (cx, cy)

def detection_thread_func(env):
    print("[Thread] Vision detection thread started.")
    while not frame_buffer.stop_flag:
        try:
            try:
                cam_name, frame_rgb = frame_buffer.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if cam_name != 'cam_tip':
                continue
                
            process_visual_feedback(env, frame_rgb, cam_name)
        except Exception as e:
            if not frame_buffer.stop_flag:
                print(f"[Thread] Error in detection: {e}")
            time.sleep(0.01)
    print("[Thread] Vision detection thread stopped.")

# ================= 主程序 =================

if __name__ == "__main__":
    env = None
    detection_thread = None

    try:
        print("Loading MuJoCo Model...")
        env = PickBoxEnv()
        
        detection_thread = threading.Thread(target=detection_thread_func, args=(env,), daemon=True)
        detection_thread.start()

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            env.reset()
            
            print("Warming up...")
            for i in range(200):
                if env.model.nu >= 6:
                    env.data.ctrl[:6] = env.data.qpos[:6]
                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                for name, cam in env.cameras.items():
                    frame = cam.get_frame()
                    if frame.size > 0:
                        test_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imshow(f"Vision: {name}", test_img)
                glfw.poll_events()
                cv2.waitKey(1)

            print("=== PHASE 1: WAITING FOR DETECTION ===")
            
            detected = False
            final_target_pos_compensated = None
            
            while not detected and viewer.is_running():
                step_start = time.time()
                
                cam_images = {}
                for name, cam in env.cameras.items():
                    cam_images[name] = cam.get_frame()
                
                if 'cam_tip' in cam_images:
                    frame_buffer.put_frame('cam_tip', cam_images['cam_tip'])
                
                if 'cam_world' in cam_images:
                    frame_bgr = cv2.cvtColor(cam_images['cam_world'], cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Vision: cam_world", frame_bgr)

                result = frame_buffer.get_result()
                if result['debug_img'] is not None:
                    cv2.imshow(f"Vision: cam_tip", result['debug_img'])
                    
                    if result['found'] and result['pixel_coords'] and not detected:
                        cx, cy = result['pixel_coords']
                        if 'cam_tip' not in env.cameras:
                            continue
                        cam_tip_obj = env.cameras['cam_tip']
                        
                        possible_z_heights = [0.04] 
                        world_pos = None
                        
                        for z_try in possible_z_heights:
                            pos_candidate = cam_tip_obj._estimate_world_pos_from_pixel(cx, cy, assumed_z=z_try, debug=False)
                            if pos_candidate is not None:
                                if -0.8 < pos_candidate[0] < 0.9 and -0.6 < pos_candidate[1] < 0.6:
                                    world_pos = pos_candidate
                                    break
                        
                        if world_pos is not None:
                            print(f"\n🎯 TARGET LOCKED! Raw Pos: {world_pos}")
                            final_target_pos_compensated = env.compensate_offset(world_pos)
                            env.enable_high_speed_mode()
                            detected = True
                
                glfw.poll_events()
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                mujoco.mj_step(env.model, env.data)
                viewer.sync()
                
                elapsed = time.time() - step_start
                if elapsed < env.dt:
                    time.sleep(max(0, env.dt - elapsed))

            if not detected:
                print("Detection failed.")
                frame_buffer.stop()
                if detection_thread:
                    detection_thread.join(timeout=1.0)
                sys.exit(0)

            frame_buffer.stop()
            if detection_thread:
                detection_thread.join(timeout=1.0)
            print("Background detection thread stopped. Switching to main-loop visual feedback.")

            print("=== STARTING HIGH-SPEED MANIPULATION ===")
            target_pos = final_target_pos_compensated
            target_euler_z = 0.0
            
            # 状态机变量
            step = 0 
            step_completed = True  # 标记当前步骤是否已完成，允许进入下一步初始化
            start_pos = None
            start_rotm = None
            start_time = 0
            target = np.zeros(6)
            
            # Step 4 专用变量
            release_triggered = False
            simulation_start_time = 0
            step4_loop_count = 0
            step4_init_done = False

            while viewer.is_running():
                step_start = time.time()
                
                # ================= 视觉反馈强制刷新 =================
                if 'cam_tip' in env.cameras:
                    frame_tip_raw = env.cameras['cam_tip'].get_frame()
                    debug_img_tip, _, _ = process_visual_feedback(env, frame_tip_raw, 'cam_tip')
                    if debug_img_tip is not None:
                        cv2.imshow(f"Vision: cam_tip", debug_img_tip)
                
                if 'cam_world' in env.cameras:
                    frame_world_raw = env.cameras['cam_world'].get_frame()
                    frame_world_bgr = cv2.cvtColor(frame_world_raw, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Vision: cam_world", frame_world_bgr)
                # =============================================================

                current_pose = env.get_current_pose()
                current_time = env.data.time

                # --- Step 0: 接近 ---
                if step == 0:
                    if step_completed:
                        print(f"\n[STEP 0 INIT] Approaching target...")
                        print(f"  Start Pos: {current_pose[0]}")
                        print(f"  Target Pos: {target_pos + np.array([0.0, 0.0, env.GRASP_APPROACH_HEIGHT])}")
                        
                        step_completed = False
                        env.need_plan = True
                        start_pos, start_rotm = current_pose[0].copy(), current_pose[1]
                        start_time = current_time
                        target[:3] = target_pos + np.array([0.0, 0.0, env.GRASP_APPROACH_HEIGHT]) 
                        target[3:] = [np.pi, 0.0, np.pi / 2]
                        print(f"  [STEP 0] Planning complete. Moving...")

                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, target[:3], target[3:], start_time, current_time
                    )
                    if done_move:
                        print(f"  [STEP 0] Completed.")
                        step = 1
                        step_completed = True

                # --- Step 1: 抓取 ---
                elif step == 1:
                    if step_completed:
                        print(f"\n[STEP 1 INIT] Moving to Grasp Height...")
                        print(f"  Start Pos: {current_pose[0]}")
                        print(f"  Target Pos: {target_pos + np.array([0.0, 0.0, env.GRASP_HEIGHT])}")
                        
                        step_completed = False
                        env.need_plan = True
                        start_pos, start_rotm = current_pose[0].copy(), current_pose[1]
                        start_time = current_time
                        target[:3] = target_pos + np.array([0.0, 0.0, env.GRASP_HEIGHT])
                        target[3:] = [np.pi, 0.0, np.pi / 2 + target_euler_z]
                        print(f"  [STEP 1] Planning complete. Moving...")

                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, target[:3], target[3:], start_time, current_time
                    )
                    if done_move:
                        print(f"  [STEP 1] Completed.")
                        step = 2
                        step_completed = True

                # --- Step 2: 抬起 ---
                elif step == 2:
                    if step_completed:
                        print(f"\n[STEP 2 INIT] Closing Gripper & Lifting...")
                        env.gripper_close()
                        # 等待夹爪闭合的物理仿真
                        for _ in range(15):
                            mujoco.mj_step(env.model, env.data)
                            viewer.sync()
                            glfw.poll_events()
                        
                        print(f"  Start Pos: {current_pose[0]}")
                        print(f"  Target Pos: {target_pos + np.array([0.0, 0.0, env.LIFT_HEIGHT])}")
                        
                        step_completed = False
                        env.need_plan = True
                        start_pos, start_rotm = current_pose[0].copy(), current_pose[1]
                        start_time = current_time
                        target[:3] = target_pos + np.array([0.0, 0.0, env.LIFT_HEIGHT])
                        target[3:] = [np.pi, 0.0, np.pi / 2]
                        print(f"  [STEP 2] Planning complete. Moving...")

                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, target[:3], target[3:], start_time, current_time
                    )
                    if done_move:
                        print(f"  [STEP 2] Completed.")
                        step = 3
                        step_completed = True

                # --- Step 3: 移动到放置区域上方 ---
                elif step == 3:
                    if step_completed:
                        print(f"\n[STEP 3 INIT] Moving to Place Location (Relative)...")
                        place_offset = np.array([0.30, 0.0, env.LIFT_HEIGHT]) 
                        place_target = target_pos + place_offset
                        print(f"  Start Pos: {current_pose[0]}")
                        print(f"  Target Pos: {place_target}")
                        
                        step_completed = False
                        env.need_plan = True
                        start_pos, start_rotm = current_pose[0].copy(), current_pose[1]
                        start_time = current_time
                        target[:3] = place_target
                        target[3:] = [np.pi, 0.0, np.pi / 2]
                        print(f"  [STEP 3] Planning complete. Moving...")

                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, target[:3], target[3:], start_time, current_time
                    )
                    if done_move:
                        print(f"  [STEP 3] Completed.")
                        step = 4
                        step_completed = True
                        # 重置 Step 4 状态
                        step4_init_done = False
                        release_triggered = False
                        step4_loop_count = 0

                # --- Step 4: 下移并检测高度释放 ---
                elif step == 4:
                    if not step4_init_done:
                        print(f"\n[STEP 4 INIT] Moving Down for Release...")
                        current_pose = env.get_current_pose() # 重新获取最新位置
                        start_pos = current_pose[0].copy()
                        start_rotm = current_pose[1]
                        start_time = current_time
                        
                        target_down_z = -0.20 
                        target_down_pos = np.array([start_pos[0], start_pos[1], target_down_z])
                        
                        print(f"  Start Pos: {start_pos}")
                        print(f"  Target Pos: {target_down_pos}")
                        print(f"  Release Threshold: {RELEASE_HEIGHT_THRESHOLD}")
                        
                        step4_init_done = True
                        step_completed = False # 运动中
                        env.need_plan = True
                        target[:3] = target_down_pos
                        target[3:] = [np.pi, 0.0, np.pi / 2]
                        print(f"  [STEP 4] Planning complete. Moving down...")

                    # 执行运动
                    done_move, curr_p, curr_e = env.line_move(
                        start_pos, start_rotm, target[:3], target[3:], start_time, current_time
                    )
                    
                    step4_loop_count += 1
                    current_z = curr_p[2]
                    
                    if step4_loop_count % 20 == 0:
                        print(f"  [STEP 4 RUN] Loop: {step4_loop_count}, Curr Z: {current_z:.4f}, Done: {done_move}")

                    # 条件 1: 高度触发释放
                    if current_z <= RELEASE_HEIGHT_THRESHOLD and not release_triggered:
                        print(f"  [SUCCESS] RELEASE TRIGGERED at Height {current_z:.4f}")
                        env.gripper_open()
                        release_triggered = True
                        
                        step = 5
                        step_completed = True # 标记 Step 4 结束，准备进入 Step 5
                        simulation_start_time = current_time
                        print(f"  [STEP 4] Transitioning to Step 5...")
                    
                    # 条件 2: 运动规划结束但未触发高度 (异常处理)
                    elif done_move and not release_triggered and step4_loop_count > 50:
                        print(f"  [ERROR] Motion completed prematurely at Z={current_z:.4f}. Forcing release.")
                        env.gripper_open()
                        release_triggered = True
                        step = 5
                        step_completed = True
                        simulation_start_time = current_time

                # --- Step 5: 自由落体仿真 ---
                elif step == 5:
                    if step_completed:
                        print(f"\n[STEP 5 INIT] Free Fall Simulation Started...")
                        step_completed = False
                        simulation_start_time = env.data.time # 确保使用最新时间

                    elapsed_sim = env.data.time - simulation_start_time
                    if elapsed_sim > SIMULATION_HOLD_TIME:
                        print("✅ Task Completed! Simulation finished.")
                        step = 6 # 结束
                    else:
                        if int(elapsed_sim * 10) % 10 == 0: # 每秒打印一次
                             pass # 可添加进度日志

                elif step >= 6:
                    print("All steps finished. Exiting loop.")
                    break

                # 事件处理
                glfw.poll_events()
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                mujoco.mj_step(env.model, env.data)
                viewer.sync()

                # 时间控制
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
        frame_buffer.stop()
        if detection_thread:
            detection_thread.join(timeout=2.0)
        
        if env:
            env.gripper_open()
            for cam in env.cameras.values():
                cam.close()
        glfw.terminate()
        cv2.destroyAllWindows()
        print("System Shutdown.")
