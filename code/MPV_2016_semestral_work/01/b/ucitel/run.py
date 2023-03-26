import cv2
import mpv01b_func

img0 = cv2.imread("../sampleData/1/0.png", cv2.IMREAD_COLOR)
img1 = cv2.imread("../sampleData/1/1.png", cv2.IMREAD_COLOR)
img2 = cv2.imread("../sampleData/1/2.png", cv2.IMREAD_COLOR)
img3 = cv2.imread("../sampleData/1/3.png", cv2.IMREAD_COLOR)

# imgs = [img0, img2, img3]
imgs = [img0, img2, img3, img1]
print "Loaded ..."

target_size = (1920, 1080)
panorama_img = mpv01b_func.panorama(imgs, target_size)

cv2.namedWindow("panorama")
cv2.imshow("panorama", panorama_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
