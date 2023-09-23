import facenet_pytorch, torch, PIL.Image, sys
infile = sys.argv[1]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = facenet_pytorch.MTCNN(device=device, selection_method='probability')
img = PIL.Image.open(infile)
boxes, probs = mtcnn.detect(img)
p = 0
if probs[0]: p = max(probs)
print(p)
