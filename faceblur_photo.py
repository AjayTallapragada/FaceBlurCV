#imports opencv2 anf mediapipe libraries for face detection and blurring
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection from the mediapipe solutions library
mp_face_detection = mp.solutions.face_detection

# Read image
img = cv2.imread('group.jpg')

#changes the image to rgb scale 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#we select model 1 for better face detection and 0.5 as min confidence as the model should have 50% confidence to move forward with detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Process the image and detect faces
    results = face_detection.process(img_rgb)

#checks if it has detected any faces in the image or it is empty
    if results.detections:
        #loops through the detected faces
        for detection in results.detections:
            # Draw  box around the face
            bboxC = detection.location_data.relative_bounding_box
            #ih- image height, iw - image width
            ih, iw, _ = img.shape
            #calculating box values
            #x- left edge, y- top edge, w- width, h- height
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Ensure values are within image bounds
            x, y = max(0, x), max(0, y)
            w, h = min(iw - x, w), min(ih - y, h)

            # Draw rectangle around the face
            face = img[y:y+h, x:x+w]
            # Apply Gaussian blur to the face region
            #99-pixel size 30- strength of the blur
            blurred = cv2.GaussianBlur(face, (99, 99), 30)
            #replace the original image with the blurred face
            img[y:y+h, x:x+w] = blurred

#create a window to display the image
cv2.imshow("Blurred Faces - Mediapipe", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
