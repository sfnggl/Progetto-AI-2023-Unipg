import os
import numpy as np
import cv2 as cv
from PIL import Image, ImageOps, ImageDraw
from scipy import ndimage
from google.colab.patches import cv2_imshow # for image display
import matplotlib.pyplot as plt
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=1.0):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true,});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

def boxes(orig):
    img = ImageOps.grayscale(orig)
    im = np.array(img)

    # External gradient
    im = ndimage.morphological_gradient(im, (5, 5))
    cv2_imshow(im)

    # Binarize
    mean, std = im.mean(), im.std()
    t = mean + std
    im[im < t] = 0
    im[im >= t] = 1

    # Connected components
    lbl, numcc = ndimage.label(im)

    # Size threshold
    min_size = 200 # pixels adjust manually
    box = [] # array box

    for i in range(1, numcc + 1):
        py, px = np.nonzero(lbl == i)
        if len(py) < min_size:
            im[lbl == i] = 0
            continue

        xmin, xmax, ymin, ymax = px.min(), px.max(), py.min(), py.max()

        # Four corners and centroid.
        box.append([
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)],
            (np.mean(px), np.mean(py))])

    return im.astype(np.uint8) * 255, box
  
def find_table_index(box):
  areamindex = -1
  aream = 0
  for x in range(len(box)):
    w = box[x][0][1][0] - box[x][0][0][0]
    h = box[x][0][2][1] - box[x][0][0][1]

    area = w * h

    if ( aream < area):
      aream = area
      areamindex = x

  return areamindex
  
def printimg(filename, box):

  img = cv.imread("photo.jpg")

  xmin = box[0][0][0]
  ymin = box[0][0][1]
  xmax = box[0][2][0]
  ymax = box[0][2][1]

  w = xmax - xmin
  h = ymax - ymin

  #image cropped have a margin of 7.5 % becouse the box results a little bit restrictive for the letters
  cropped_image = img[ int(ymin - h*0.075) : int(ymax + h*0.075) , int(xmin - w*0.075) : int(xmax + w*0.075)]

  # Convert to grayscale
  gray = cv.cvtColor(cropped_image, cv.COLOR_BGR2GRAY)

  # Blur the image
  blur = cv.GaussianBlur(gray, (5, 5), 0)

  th_GAUSSIAN = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

  #resize image
  resize = cv.resize(th_GAUSSIAN, (28, 28), interpolation = cv.INTER_CUBIC)

  #show and save
  cv2_imshow(resize)
  cv.imwrite(filename, resize)
  
def main():
	orig = Image.open("photo.jpg")
	im, box = boxes(orig)

	# Draw perfect rectangles and the component centroid.
	img = Image.fromarray(im)
	visual = img.convert('RGB')
	draw = ImageDraw.Draw(visual)
	for b, centroid in box:
		  draw.line(b + [b[0]], fill='yellow')
		  cx, cy = centroid
		  draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill='red')
	visual.show()

	#remove table cell
	box.pop(find_table_index(box))

	#order by y center coordinate
	box.sort(key= lambda x: x[1][1])

	#define a matrix
	table = [[]]
	row = 0
	y_row_min = 0
	y_row_max = 0

	table[row].append(box[0])
	y_row_min = box[0][0][0][1]
	y_row_max = box[0][0][2][1]

	print("FIRST RAW: y min: "+str(y_row_min)+" y max: "+str(y_row_max) + "\nPrimo elemento inserito\n")

	for x in box[1:]:

		print("ACTIVE #"+str(row)+" RAW: y min: "+str(y_row_min)+" y max: "+str(y_row_max))
		print("BOX: y_center: "+str((x[0][0][1]+x[0][2][1])/2))

		if ( y_row_min*0.9 < (x[0][0][1]+x[0][2][1])/2 < y_row_max*1.1):
		  table[row].append(x)
		  if(y_row_min > x[0][0][1]):
		    y_row_min = x[0][0][1]
		  if(y_row_max < x[0][2][1]):
		    y_row_max = x[0][2][1]

		  print("Elemento inserito in riga "+str(row)+"\n")

		else:
		  table.append([])
		  row = row + 1
		  table[row].append(x)
		  y_row_min = x[0][0][1]
		  y_row_max = x[0][2][1]

		  print("Nuova riga creata e elemento inserito in riga "+str(row)+"\n")

	#sort for x coordinate (sort one raw at time)
	for x in table:
		x.sort(key= lambda x: x[1][0])

	w=len(table)
	h=len(table[0])

	print("La tabella è di dimensione: "+str(w)+"x"+str(h)+"\n")

	#save image
	final_arr=[]
	for row in range(len(table)):
		for col in range(len(table[row])):
		  filename = str(row)+str(col)+".jpg"
		  print(filename)
		  print(table[row][col])
		  printimg(filename, table[row][col])
		  final_arr.append(cv.imread(filename))
  
	#final print
	for i in range(len(box)):
		plt.subplot(w,h,i+1)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.imshow(final_arr[i])
		#plt.xlabel(class_names[label])
	plt.show()

	if os.path.isfile("photo.jpg"):
		os.remove("photo.jpg")

	# Check if directory "output" exists and remove it recursively
	if os.path.exists("output"):
		for filename in os.listdir("output"):
			file_path = os.path.join("output", filename)
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				elif os.path.isdir(file_path):
					os.rmdir(file_path)
			except Exception as e:
				print(f"Failed to delete {file_path}. Reason: {e}")
		os.rmdir("output")

	os.makedirs("output")

	for filename in os.listdir():
		if filename.endswith(".jpg"):
			os.rename(filename, os.path.join("output", filename))
