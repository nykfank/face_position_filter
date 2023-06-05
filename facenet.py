import facenet_pytorch, torch, os, PIL.Image, math
indir = '/home/nyk/test_images'
indir = '/mnt/big/nick/cams/eingang'
target_pos = 775, 250
target_size = 2800
logfn = 'eingang_facenet.txt'
open(logfn, 'w')

def logg(x):
	print(x)
	open(logfn, 'a').write(x + '\n')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = facenet_pytorch.MTCNN(device=device, selection_method='probability')
# {'box': [186, 71, 87, 115], 'confidence': 0.9994562268257141, 'keypoints': {'left_eye': (207, 110), 'right_eye': (252, 119), 'nose': (220, 143), 'mouth_left': (200, 148), 'mouth_right': (244, 159)}}

for f in os.listdir(indir):
	if not f.endswith('.jpg'): continue
	img = PIL.Image.open('%s/%s' % (indir, f))
	#boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
	boxes, probs = mtcnn.detect(img)
	if not probs[0]:
		logg('%s, %s' % (f, probs[0]))
		continue
	for box, prob in zip(boxes, probs):
		if prob < 0.9:
			logg('%s, %f' % (f, prob))
			continue
		box_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
		box_dist = math.sqrt((box_center[0] - target_pos[0]) ** 2 + (box_center[1] - target_pos[1]) ** 2)
		area = (box[2] - box[0]) * (box[3] - box[1])
		area_dist = abs(target_size - area)
		logg('%s, %f, %d/%d/%d/%d, %d, %d' % (f, probs[0], box[0], box[1], box[2], box[3], box_dist, area_dist))
		if box_dist > 60: continue
		if area_dist > 600: continue
		# Filter auf Augenabstand? Oder wie am besten nicht-frontal erkennen?
		face = img.crop(box)
		ofn = 'output/%s.png' % os.path.splitext(f)[0]
		face.save(ofn)
		ofn2 = 'output2/%s' % f
		img.save(ofn2)

