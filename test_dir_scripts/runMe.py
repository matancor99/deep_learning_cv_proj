
# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys



def run(myAnnFileName, buses):
	annFileNameGT = os.path.join(os.getcwd(),'annotationsTest.txt')
	writtenAnnsLines = {}
	annFileEstimations = open(myAnnFileName, 'w+')
	annFileGT = open(annFileNameGT, 'r')
	writtenAnnsLines['Ground_Truth'] = (annFileGT.readlines())

	TEST_IMAGES_DIR = 'images'
	MODEL_NAME = 'inference_graph'

	strToWrite = ''


	# Grab path to current working directory
	CWD_PATH = os.getcwd()

	# Path to frozen detection graph .pb file, which contains the model that is used
	# for object detection.
	PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')




	# Number of classes the object detector can identify
	NUM_CLASSES = 6

	# Load the Tensorflow model into memory.
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

		sess = tf.Session(graph=detection_graph)

	# Define input and output tensors (i.e. data) for the object detection classifier

	# Input tensor is the image
	image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

	# Output tensors are the detection boxes, scores, and classes
	# Each box represents a part of the image where a particular object was detected
	detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

	# Each score represents level of confidence for each of the objects.
	# The score is shown on the result image, together with the class label.
	detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
	detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

	# Number of objects detected
	num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        
	for k, line_ in enumerate(writtenAnnsLines['Ground_Truth']):
		line = line_.replace(' ','')
		imName = line.split(':')[0]

		# Name of the directory containing the object detection module we're using

		IMAGE_NAME = imName

		# Path to image
		PATH_TO_IMAGE = os.path.join(CWD_PATH, TEST_IMAGES_DIR, IMAGE_NAME)

		# Load image using OpenCV and
		# expand image dimensions to have shape: [1, None, None, 3]
		# i.e. a single-column array, where each item in the column has the pixel RGB value
		image = cv2.imread(PATH_TO_IMAGE)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image_expanded = np.expand_dims(image, axis=0)

		# Perform the actual detection by running the model with the image as input
		(boxes, scores, classes, num) = sess.run(
			[detection_boxes, detection_scores, detection_classes, num_detections],
			feed_dict={image_tensor: image_expanded})


		min_score_thresh=0.80
		im_height, im_width, im_channels = image.shape

		boxes = boxes[0]
		scores = scores[0]
		classes = classes[0]
		num = num[0]

		new_boxes = []
		new_scores = []
		new_classes = []
		new_box_class = []
		new_num = 0
		for index in range(len(boxes)):
			if scores[index] > min_score_thresh:
				#ymin, xmin, ymax, xmax
				box = boxes[index]

				non_normalized_box = [int(box[0] * im_height), int(box[1] * im_width), int(box[2] * im_height), int(box[3] * im_width)]

				new_boxes.append(non_normalized_box)
				new_scores.append(scores[index])
				new_classes.append(classes[index])

				# xmin, ymin, width, hight, color
				box_class = [non_normalized_box[1] , non_normalized_box[0], non_normalized_box[3] - non_normalized_box[1], non_normalized_box[2] - non_normalized_box[0], int(classes[index])]
				new_box_class.append(box_class)
				new_num += 1

		#print(new_box_class)

		ItstrToWrite = IMAGE_NAME + ':' + str(new_box_class) + '\n'
		ItstrToWrite = ItstrToWrite.replace(' ','')
		ItstrToWrite = ItstrToWrite.replace('[[','[')
		ItstrToWrite = ItstrToWrite.replace(']]',']')
		strToWrite += ItstrToWrite
		print(ItstrToWrite)

	print(strToWrite)
	annFileEstimations.write(strToWrite)
