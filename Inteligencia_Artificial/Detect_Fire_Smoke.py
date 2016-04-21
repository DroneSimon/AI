import cv2
import numpy as np
import xml.etree.ElementTree as ET

class Detect_Pattern_Fire_Smoke():

    def __init__(self):
        self.TRAIN_FIRE = 'Train/Fire/COLOR_F-FP_17.5%.xml'
        self.TRAIN_SMOKE = 'Train/Smoke/COLOR_H-FP_14.29%.xml'
        # The colors are:blue,red,green
        self.ARRAY_COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
        self.MAX_SMOKE = -20.0
        self.MAX_FIRE = -20.0

    def detect_fire(self, image):
        if image!= None:
            recognizer = self.load_train_fire(image)
        else:
            print 'The image is not valid'
            recognizer = None
        return recognizer

    def detect_smoke(self, image):
        if image != None:
            recognized_image = self.recognize_smoke(image)
        else:
            print 'The image is not valid'
            recognized_image = None
        return recognized_image

    def detect_fire_smoke(self, image):
        list_recognizer = None
        if image != None:
            #get a list of the two types of recognized groups [fire, smoke]
            list_recognizer = self.recognize(image)
        else:
            print "The path of the image is not correct"
        return list_recognizer

    def recognize(self, image):
        recognizers = []
        recognizer = self.recognize_smoke(image)
        recognizers.append(recognizer)
        recog_fire = self.load_train_fire(image)
        recognizers.append(recog_fire)
        return recognizers

    def recognize_smoke(self, image):
        recognizers = []
        hog = self.load_train_smoke(self.TRAIN_SMOKE)
        sections, weights = hog.detectMultiScale(image, hitThreshold=2.0, scale=1.5)
        if len(sections) == 0:
            print "The train is not found"
        else:
            self.MAX_SMOKE = max(weights)
            weights = list(weights)
            index_max_smoke = weights.index(self.MAX_SMOKE)
            res_max = sections[index_max_smoke]
            x,y,w,h = res_max
            rect = [x,y,x+w,y+h]
            recognizers.append(rect)
            recognizers = np.array(recognizers)
        return recognizers

    def load_train_smoke(self, path_file):
        tree = ET.parse(path_file)
        root = tree.getroot()
        vectors = root[0].find('support_vectors')
        vectors = vectors[0].text
        SV = [np.float32(line) for line in vectors.split()]
        labels = root[0].find('decision_functions')[0].find('rho')
        SV.append(-np.float32(labels.text))
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        hog.setSVMDetector(np.array(SV))
        return hog

    def load_train_fire(self, image):
        rec_fire = []
        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        model = cv2.SVM()
        mat_points= self.map_out(v.astype(np.uint8), image)
        sorter = self.TRAIN_FIRE
        model.load(sorter)
        hog = cv2.HOGDescriptor((64,64), (16,16), (8,8),(8,8),9)
        for (x,y,w,h) in mat_points:
            subMat = image[y:h, x:w]
            subMat = cv2.resize(subMat,(64,64))
            descriptor = hog.compute(subMat)
            descriptor = np.concatenate(descriptor)
            res = model.predict(descriptor)
            if res > self.MAX_FIRE:
                self.MAX_FIRE = res
                rect = (x,y,w,h)
                rec_fire.append(rect)
        return rec_fire

    def map_out(self, mat_v, image):
        mat_points = []
        #mat_v = cv2.equalizeHist(mat_v)
        ret, binary_image = cv2.threshold(mat_v, 240, 255,cv2.THRESH_BINARY)

        contours, inheriters = cv2.findContours(binary_image,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            moments = cv2.moments(c)
            if(moments['m00']>50):
                x = []
                y = []
                for i in c:
                    for j in i:
                        x.append(j[0])
                        y.append(j[1])
                max_x, min_x, max_y, min_y = np.argmax(x), np.argmin(x), np.argmax(y), np.argmin(y)
                mat_points.append((x[min_x],y[min_y],x[max_x],y[max_y]))

        return mat_points

    def apply_camshift(self, list_rect, image):
        copy = image.copy()
        i = 0
        for group in list_rect:
            for (x,y,w,h) in group:
                track_window = (x,y,w,h)
                # set up the ROI for tracking
                roi = image[y:y + h , x:x + w]
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
                roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
                cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
                # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
                term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
                # apply mean shift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                # Draw it on image
                x,y,w,h = track_window
                cv2.rectangle(copy, (x,y), (x+w,y+h), self.ARRAY_COLORS[i], 2)
            i+=1
        return copy

#The variable order defines who is going to be detect: f(detect_fire), s(detect_smoke), f_s(detect fire and smoke)
order = "f_s"
detect_pattern = Detect_Pattern_Fire_Smoke()
#Must change by the path of video that want to analize
path_video = 'Videos/scena15.mp4'
capture = cv2.VideoCapture(path_video)
#If you desired grab the video, must uncomment the next line, modifying the variable fourcc with the suitable codec by your PC.
#video = cv2.VideoWriter("Videos/video_test_camshift_scena15.avi", fourcc=cv2.cv.CV_FOURCC('m','p','4','v'),fps=10,frameSize=(640,480))
frame_res = None
while True:
    detected_frame = None
    ret, frame = capture.read()
    frame = cv2.resize(frame,(640,480))
    if(order == "f"):
        detected_frame = [detect_pattern.detect_fire(frame)]
    elif(order == "s"):
        detected_frame = [detect_pattern.detect_smoke(frame)]
    elif(order == "f_s"):
        detected_frame = detect_pattern.detect_fire_smoke(frame)
    if(detected_frame is not None):
        if frame_res is None:
            frame_res = detected_frame

        image_camshift = detect_pattern.apply_camshift(frame_res, frame)
        #This line also must uncomment to grab the video
        #video.write(image_camshift)
        cv2.imshow("camshift", image_camshift)
    k = cv2.waitKey(1)
    if k == 27:
        break
print "The higher percentage of fire: " + str(detect_pattern.MAX_FIRE)
print "The higher percentage of smoke: " + str(detect_pattern.MAX_SMOKE)
video.release()
cv2.destroyAllWindows()