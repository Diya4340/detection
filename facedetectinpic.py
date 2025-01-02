import cv2

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_in_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Resize the image for display (optional, set max dimensions)
    max_width, max_height = 800, 600  # Adjust based on your display resolution
    h, w = image.shape[:2]
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the result
    cv2.imshow("Faces Detected", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detect_faces_in_image('diya.jpeg')  # Replace with your image file path
