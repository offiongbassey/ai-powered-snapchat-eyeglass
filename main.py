import cv2
import mediapipe as mp
import os 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

eyeglass = cv2.imread(os.path.join('.', 'glass.png'), cv2.IMREAD_UNCHANGED)

webcam = cv2.VideoCapture(0)

def overlay_filter(frame, filter_img, left_eye, right_eye):
    mid_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    filter_width = int(abs(right_eye[0] - left_eye[0]) * 2)
    filter_height = int(filter_width * filter_img.shape[0] // filter_img.shape[1])

    filter_resize = cv2.resize(filter_img, (filter_width, filter_height))

    #Get Positions or cordinates
    x1 = mid_eye[0] - filter_width // 2
    y1 = mid_eye[1] - filter_height // 2
    x2 = x1 + filter_width
    y2 = y1 + filter_height

    if x1 >=0 and y1 >=0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
        alpha_s = filter_resize[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(3):
            frame[y1:y2, x1:x2, c] = (alpha_s * filter_resize[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
        
    return frame

while webcam.isOpened:
    ret, frame = webcam.read()

    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(img_rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye = (int(face_landmarks.landmark[33].x * w), int(face_landmarks.landmark[33].y * h))
            right_eye = (int(face_landmarks.landmark[263].x * w), int(face_landmarks.landmark[263].y * h))

            frame = overlay_filter(frame, eyeglass, left_eye, right_eye)
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()