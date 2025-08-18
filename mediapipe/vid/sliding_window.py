import cv2
import mediapipe as mp
import json
import math  # 引入math库来进行向上取整

# --- MediaPipe 初始化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_settings = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- 1. 参数设置 (Parameter Setup) ---
# 在这里集中管理你的参数，方便修改
WINDOW_SECONDS = 60  # 窗口大小（秒）
STEP_SECONDS = 5  # 步进/滑动大小（秒）
VID_PATH = 'vid.mp4'  # 视频路径

# 初始化 Face Mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,  # 设置为False来处理视频
    max_num_faces=1,
    refine_landmarks=True,  # 设置为True可以获取更精确的嘴唇、眼睛、瞳孔的特征点
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- 2. 获取视频属性并计算帧数 (Get Video Properties & Calculate Frames) ---
capture = cv2.VideoCapture(VID_PATH)
if not capture.isOpened():
    print(f"Error: Could not open video file: {VID_PATH}")
    exit()

# 获取视频的FPS和总帧数
fps = capture.get(cv2.CAP_PROP_FPS)  # 帧率（Frames Per Second）
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))   # 总帧数（Total Frames）

# 检查FPS和总帧数是否有效
if fps == 0 or total_frames == 0:
    print("Error: Video file might be corrupted or FPS/frame count is zero.")
    capture.release()
    exit()

# 将时间和步进从秒转换为帧
window_in_frames = int(WINDOW_SECONDS * fps)
step_in_frames = int(STEP_SECONDS * fps)

print(f"Video Info: FPS={fps:.2f}, Total Frames={total_frames}")
print(f"Windowing Info: Window Size={window_in_frames} frames, Step Size={step_in_frames} frames")

# --- 3. 主循环 - 按窗口处理 (Main Loop - Process by Window) ---
all_windows_data = []
window_count = 0

# 外层循环：根据步进大小移动窗口
# range(start, stop, step)
for start_frame in range(0, total_frames, step_in_frames):
    end_frame = start_frame + window_in_frames
    # 确保结束帧不会超出视频总长度
    if end_frame > total_frames:
        end_frame = total_frames

    print(f"\nProcessing Window {window_count + 1}: Frames [{start_frame} -> {end_frame - 1}]")

    # 当前窗口的数据存储
    current_window_landmarks = {}

    # 将视频读取指针设置到当前窗口的起始位置
    # 这是实现窗口处理的关键步骤！
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 内层循环：处理当前窗口内的每一帧
    # current_frame_index 从 start_frame 开始计数
    for current_frame_index in range(start_frame, end_frame):
        success, image = capture.read()
        if not success:
            # 如果读不到帧，说明可能已经到视频末尾，跳出内层循环
            print(f"Warning: Could not read frame {current_frame_index}. Reached end of video.")
            break

        # 将图像从BGR转换为RGB，因为MediaPipe需要RGB格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # 提高性能

        # 使用MediaPipe处理图像
        results = face_mesh.process(image_rgb)

        # 提取并存储特征点
        if results.multi_face_landmarks:
            # 假设只处理视频中的第一张脸
            landmarks_for_this_frame = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in results.multi_face_landmarks[0].landmark
            ]
            current_window_landmarks[current_frame_index] = landmarks_for_this_frame

        # --- 可选：实时显示处理结果 ---
        # image_rgb.flags.writeable = True
        # image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # if results.multi_face_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image=image_bgr,
        #         landmark_list=results.multi_face_landmarks[0],
        #         connections=mp_face_mesh.FACEMESH_TESSELATION,
        #         landmark_drawing_spec=drawing_settings,
        #         connection_drawing_spec=drawing_settings)
        # cv2.imshow('Processing Window', image_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 将当前窗口处理好的数据添加到总列表中
    if current_window_landmarks:  # 只有当窗口内提取到数据时才添加
        all_windows_data.append({
            'window_id': window_count,
            'start_frame': start_frame,
            'end_frame': end_frame - 1,  # 记录实际处理的最后一帧
            'landmarks': current_window_landmarks
        })
    window_count += 1
    # 如果用户按了 'q' 键，也要跳出外层循环
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# --- 4. 保存结果 (Save Results) ---
output_filename = 'landmarks_by_window.json'
with open(output_filename, 'w') as f:
    json.dump(all_windows_data, f, indent=4)

print(f"\nProcessing complete. Data saved to {output_filename}")

# --- 5. 清理 (Cleanup) ---
face_mesh.close()
capture.release()
cv2.destroyAllWindows()