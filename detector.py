import numpy as np
import argparse
import time
import cv2
import os
import color_classifier
import sqlite3
import io
import requests
from PIL import Image
conn = sqlite3.connect(".\plate.db")
cur = conn.cursor()
def extract_features(frame):
    regions=['fr','it']
    imgbytes=io.BytesIO()
    img=Image.fromarray(frame)
    img.save(imgbytes,'jpeg')
    imgbytes.seek(0)
    response=requests.post(
        'https://api.platerecognizer.com/v1/plate-reader/',
        data=dict(regions=regions),
        files=dict(upload=imgbytes),
        headers={'Authorization':'Token bc62b61ef26409e4e235f2ff0dc66a3800a6ff37'})
    final_result=response.json()
    return final_result


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", default='yolo-coco',
                help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

car_color_classifier = color_classifier.Classifier()
#car_model_classifier = model_classifier.Classifier()

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(args["input"])
vid_writer=cv2.VideoWriter(args['output'],
                           cv2.VideoWriter_fourcc('M','J','P','G'),20,
                           (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
c=0
result={}
while(True):
    ret,image=cap.read()
    H=W=0
    if ret == True:
        (H,W)=image.shape[:2]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        outputs = net.forward(output_layers)
        end = time.time()

        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []
        areas=[]

        for output in outputs:
            for detection in output:
                scores = detection[5:]#extract confidence and classID
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    #Scaling to original image size
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    area=0.5*width*height
                    areas.append(area)
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        print(boxes,confidences,classIDs,areas)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                if classIDs[i] == 2:
                    start = time.time()
                    color_result = car_color_classifier.predict(image[max(y,0):y + h, max(x,0):x + w])
                    end = time.time()
                    print("[INFO] classifier took {:.6f} seconds".format(end - start))
                    text = "{}: {:.4f}".format(color_result[0]['color'], float(color_result[0]['prob']))
                    cv2.putText(image, text, (x + 2, y + 20) ,cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,247,255), 2,cv2.LINE_AA)

                    if c%5 == 0:
                        result = extract_features(image)
                        print("done")
                        if len(result['results'])>0:
                            n_plate = result['results'][0]['plate'].upper()
                            print(n_plate,type(n_plate))

                            cur.execute("Select * from number_plate where numberplate='{}'".format(str(n_plate)))
                            data = cur.fetchall()
                            print(data,len(data))
                            if len(data)>0:
                                #print("ALLOWED")
                                cv2.putText(image,"ALLOWED",(x+100,y+200),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,247,255),2,cv2.LINE_AA)
                            else:
                                #print("NOT ALLOWED")
                                cv2.putText(image,"NOT ALLOWED",
                                            (x+100,y+200),
                                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,247,255),2,cv2.LINE_AA)
                    c+=1

                cv2.rectangle(image, (x, y), (x + w, y + h), (0,247,255), 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x+2, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,247,255), 2,cv2.LINE_AA)

        cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image', W, H)
        cv2.imshow("Image", image)
        vid_writer.write(image.astype(np.uint8))
        if cv2.waitKey(1)==27:
            break
cv2.destroyAllWindows()

