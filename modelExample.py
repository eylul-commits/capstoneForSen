from getFaceData import FaceModel
import cv2

model = FaceModel()
img = cv2.imread("kid female.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(model.getFaceData(img))

cv2.waitKey(0)