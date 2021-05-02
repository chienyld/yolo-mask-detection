import argparse
import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.5, help='Confidence for yolo')
args = ap.parse_args()

classes = ["good", "bad", "none"]

if args.network == "normal":
    print("loading yolov4...")
    yolo = YOLO("models/mask-yolov4.cfg", "models/mask-yolov4.weights", classes)
elif args.network == "prn":
    print("loading yolov3-tiny-prn...")
    yolo = YOLO("models/mask-yolov3-tiny-prn.cfg", "models/mask-yolov3-tiny-prn.weights", classes)
else:
    print("loading yolov4-tiny...")
    yolo = YOLO("models/mask-yolov4-tiny.cfg", "models/mask-yolov4-tiny.weights", classes)

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255)]

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    width, height, inference_time, results = yolo.inference(frame)
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        color = colors[id]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    serious = set()
    abnormal = set()

	# ensure there are *at least* two people detections (required in
	# order to compute our pairwise distance maps)
    if len(results) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number of pixels
                if D[i, j] < config.MIN_DISTANCE:
					# update our violation set with the indexes of the centroid pairs
                    serious.add(i)
                    serious.add(j)
                # update our abnormal set if the centroid distance is below max distance limit
				if (D[i, j] < config.MAX_DISTANCE) and not serious:
					abnormal.add(i)
					abnormal.add(j)

	# loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
		# extract the bounding box and centroid coordinates, then
		# initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation/abnormal sets, then update the color
        if i in serious:
            color = (0, 0, 255)
        elif i in abnormal:
            color = (0, 255, 255) #orange = (0, 165, 255)

        # draw (1) a bounding box around the person and (2) the
        # centroid coordinates of the person,
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 2)
    
        seriousText = "Total serious violations: {}".format(len(serious))
        cv2.putText(frame, seriousText, (10, frame.shape[0] - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

        abnormalText = "Total abnormal violations: {}".format(len(abnormal))
        cv2.putText(frame, abnormalText, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    cv2.imshow("preview", frame)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
