import numpy as np
import cv2
def detected(images):
    img = []
    for a in range(len(images)):
        print(a)
        image = images[a]
        image = cv2.resize(image,[64,64])
        # with open('Classes.txt', 'r') as f:
        #     class_labels = f.read().splitlines()
        # # Load the YOLOv3 weights and configuration
        # net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        # # Obtain the input dimensions of the network
        # layer_names = net.getLayerNames()
        # output_layers = [layer_names[-1] for i in net.getUnconnectedOutLayers()]
        # height, width, channels = image.shape
        # # Preprocess the image
        # blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)
        # # Set the input blob for the network
        # net.setInput(blob)
        # # Run the forward pass to get the output layers
        # outs = net.forward(output_layers)
        # # Set the confidence threshold and non-maximum suppression threshold
        # conf_threshold = 0.5
        # nms_threshold = 0.4
        # # Process the output layers
        # class_ids = []
        # confidences = []
        # boxes = []
        # for out in outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > conf_threshold:
        #             center_x = int(detection[0] * width)
        #             center_y = int(detection[1] * height)
        #             w = int(detection[2] * width)
        #             h = int(detection[3] * height)
        #             # Calculate the top-left corner coordinates of the bounding box
        #             x = int(center_x - w / 2)
        #             y = int(center_y - h / 2)
        #             # Collect the bounding box, class ID, and confidence
        #             boxes.append([x, y, w, h])
        #             class_ids.append(class_id)
        #             confidences.append(float(confidence))
        # # Apply non-maximum suppression to remove overlapping bounding boxes
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        # # Draw the bounding boxes and labels on the image
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # if len(indices) > 0:
        #     for i in indices.flatten():
        #         x, y, w, h = boxes[i]
        #         label = class_labels[class_ids[i]]
        #         confidence = confidences[i]
        #         # Draw the bounding box rectangle and label
        #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #         cv2.putText(image, f'{label}: {confidence:.2f}', (x, y - 5), font, 0.5, (0, 255, 0), 1)
        # cv2.imshow('jwhjc',image)
        # cv2.waitKey(0)
        img.append(image)
    return img
