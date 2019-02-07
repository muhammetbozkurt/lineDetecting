import cv2
import numpy as np

def line_detect(image,treshold_l=120,treshold_h=200):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h,w = gray_image.shape
    gaussian_image = cv2.GaussianBlur(gray_image,(5,5),0)
    canny_image = cv2.Canny(gaussian_image,treshold_l,treshold_h)
    ROI = np.array([[(0,h-20),(330,240),(370,240),(w,h-20)]],dtype=np.int32)#niye çift array i
    blank = np.zeros_like(canny_image)
    mask = cv2.fillPoly(blank,ROI,255)
    masked_image = cv2.bitwise_and(canny_image,mask)
    lines = cv2.HoughLines(canny_image,1,np.pi/360,300)
    copy_im = np.zeros((h,w,3))
    if lines is not None:
        for i in range(0,len(lines)):
            ro = lines[i][0][0]
            te = lines[i][0][1]

            x0 = ro * np.cos(te)
            y0 = ro * np.sin(te)

            a = np.cos(te)
            b = np.sin(te)

            x1 = int(x0+1000 * (-b))
            y1 = int(y0+1000 * (a))
            x2 = int(x0-1000 * (-b))
            y2 = int(y0-1000 * (a))
            cv2.line(copy_im,(x1,y1),(x2,y2),255,2)

    return copy_im

def line_detectp(image,treshold_l=120,treshold_h=200):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	h,w = gray_image.shape
	gaussian_image = cv2.GaussianBlur(gray_image,(13,13),0) #orjinali 5,5
	canny_image = cv2.Canny(gaussian_image,treshold_l,treshold_h)
	ROI = np.array([[(0,h-20),(w/2-25,h/2+75),(w/2+25,h/2+75),(w,h-20)]],dtype=np.int32)#niye çift array i
	blank = np.zeros_like(canny_image)
	mask = cv2.fillPoly(blank,ROI,255)
	masked_image = cv2.bitwise_and(canny_image,mask)
	#lines = cv2.HoughLinesP(image,rho ,theta ,threshold ,minLineLength ,maxLineGap )
	lines = cv2.HoughLinesP(masked_image,2,np.pi/1080,30,30,5)
	copy_im = np.zeros((h,w,3))
	if lines is not None:
		for values in lines:
			cv2.line(copy_im,(values[0][0],values[0][1]),(values[0][2],values[0][3]),(0,0,255),2)
		#copy_im=cv2.addWeighted(copy_im,1,image,1,0)
	return copy_im

def mask_try(image):
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	h,w =gray_image.shape

	ROI = np.array([[(0,h-20),(w/2-25,h/2+75),(w/2+25,h/2+75),(w,h-20)]],dtype=np.int32)#niye çift array i 
	blank = np.zeros_like(gray_image)
	mask = cv2.fillPoly(blank,ROI,255)
	masked_image = cv2.bitwise_and(gray_image,mask)
	return masked_image

def canny(image):
	threshold1,threshold2 = 80,150
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gaussian_image = cv2.GaussianBlur(gray,(5,5),0)
	canny = cv2.Canny(gray, threshold1,threshold2)
	return canny

def laplacian(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(5,5),0)
	laplacian = cv2.Laplacian(gray,cv2.CV_64F)
	return laplacian

image = cv2.imread("yol_viraj.jpg")
result = line_detect(image)
resultp = line_detectp(image)
cv2.imshow("deneme",image)
cv2.imshow("lines",result) 
cv2.waitKey()
cv2.destroyAllWindows()


cap = cv2.VideoCapture("video.mp4")

while True:
	ret, frame = cap.read()
	if ret is True:
		cv2.imshow("line_detectp",line_detectp(frame,treshold_l=80,treshold_h=120))
		#cv2.imshow("laplacian",laplacian(frame))
		if cv2.waitKey(1) == 13:
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()