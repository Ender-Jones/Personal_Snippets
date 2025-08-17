import cv2
import mediapipe as mp
import json

# the test is for mediapipe vid face detection study.
# # detection and tracking using mediapipe face mesh
# # import drawing_utils for drawing landmarks
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# default settings for face mesh
drawing_settings = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# video path
# # if you want to use webcam, set vid_path = 0
# # cv2.VideoCapture(path) will open the video file or webcam
vid_path = 'vid.mp4'
capture = cv2.VideoCapture(vid_path)

#  initialize face mesh with default parameters
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# result for all landmarks
landmarks_per_frame = {}
# frame counter
frame_counter = 0

# check if the video is opened successfully
if not capture.isOpened():
    print("Error: Could not open video.")
else:
    while capture.isOpened():
        success, image = capture.read()
        if not success:
            print("Error: Could not read frame.")
            break

        # convert the image to RGB
        # if you are using webcam, you might want to flip the image by cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # to boost performance, mark the image as not writeable
        image.flags.writeable = False
        results = face_mesh.process(image)

        # add the landmarks to the list
        if results.multi_face_landmarks:
            landmarks_for_this_frame = [
                {'x': lm.x, 'y': lm.y, 'z': lm.z}
                for lm in results.multi_face_landmarks[0].landmark
            ]
            landmarks_per_frame[frame_counter] = landmarks_for_this_frame
        frame_counter += 1

        # mark the image as writeable again
        image.flags.writeable = True

        # convert the image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for landmarks_per_face in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=landmarks_per_face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_settings,
                    connection_drawing_spec=drawing_settings)

            cv2.imshow('Mediapipe Face Mesh', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    with open('landmarks_per_frame.txt', 'w') as f:
        f.write(json.dumps(landmarks_per_frame, indent=4))

    # cleanup
    face_mesh.close()
    cv2.destroyAllWindows()
