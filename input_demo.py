import mujoco
import time
import glfw
import numpy as np
import cv2
from pick_box_env_input import PickBoxEnv

def get_user_input():
    """获取用户输入的抓取和放置颜色"""
    print("\n=== UR5 抓取任务 ===")
    print("可用颜色: red (红色), green (绿色), blue (蓝色)")
    
    while True:
        try:
            pick_color = input("请输入要抓取的颜色 (red/green/blue): ").strip().lower()
            if pick_color not in ['red', 'green', 'blue']:
                print("无效颜色，请输入 red, green 或 blue")
                continue
                
            place_color = input("请输入要放置到的颜色位置 (red/green/blue): ").strip().lower()
            if place_color not in ['red', 'green', 'blue']:
                print("无效颜色，请输入 red, green 或 blue")
                continue
                
            if pick_color == place_color:
                print("抓取和放置的颜色不能相同，请重新选择")
                continue
                
            return pick_color, place_color
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            return None, None

def main():
    env = PickBoxEnv()
    
    # 获取用户输入
    pick_color, place_color = get_user_input()
    if pick_color is None:
        return
    
    print(f"\n开始执行任务: 从 {pick_color} 抓取，放置到 {place_color}")
    
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # 使用用户指定的颜色重置环境
        env.reset(pick_color=pick_color, place_color=place_color)
        start_pos, start_rotm = env.get_current_pose()

        done = False
        cnt = -1
        step = -1
        step_res = True

        # 从配置获取抓取参数
        grasp_config = env.config['grasping']
        approach_height = grasp_config['approach_height']
        grasp_height = grasp_config['grasp_height']
        lift_height = grasp_config['lift_height']
        place_height = grasp_config['place_height']

        while viewer.is_running() and not done:
            cnt += 1
            step_start = time.time()

            if step == -1 and step_res:
                step = 0
                step_res = False
                env.need_plan = True
                start_pos, start_rotm = env.get_current_pose()
                box_pos, box_eulerz = env.get_box_pos(env.pick_box)
                target = [0.0, 0.0, 0.0, np.pi, 0.0, np.pi / 2]
                target[:3] = box_pos + [0.0, 0.0, approach_height]
                start_time = env.data.time

            elif step == 0 and step_res:
                step = 1
                step_res = False
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, grasp_height]
                target[3:] = [np.pi, 0.0, np.pi / 2 + box_eulerz]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 1 and step_res:
                env.gripper_close()
                step = 2
                step_res = False
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, lift_height]
                target[3:] = [np.pi, 0.0, np.pi / 2]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 2 and step_res:
                step_res = False
                step = 3
                env.need_plan = True
                box_pos, box_eulerz = env.get_box_pos(env.place_box)
                target[:3] = box_pos + [0.0, 0.0, approach_height]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time

            elif step == 3 and step_res:
                step_res = False
                step = 4
                env.need_plan = True
                target[:3] = box_pos + [0.0, 0.0, place_height]
                target[3:] = [np.pi, 0.0, np.pi / 2 + box_eulerz]
                start_pos, start_rotm = env.get_current_pose()
                start_time = env.data.time
                
            elif step == 4 and step_res:
                env.gripper_open()
                done = True
                print("任务完成！")
            else:
                pass

            step_res, action_pos, action_euler = env.line_move(
                start_pos, start_rotm, target[:3], target[3:], start_time, env.data.time
            )
            
            if cnt % 20 == 0:
                env.cam_tip.show_img()
                env.cam_world.show_img()

            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            
            # 时间控制
            if env.config['simulation']['sleep_control']:
                time_until_next_step = env.dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

        # 等待用户查看结果
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        
        glfw.terminate()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
