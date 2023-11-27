import cv2 as cv
from google.colab.patches import cv2_imshow

def close_enough_linear(prob_array, labels):
  prob_array=np.asarray(prob_array)

  #= (X, 1, 4)
  if (prob_array.ndim == 3 and prob_array[0].shape == (1, 4)):
    #
    index,tmp = -1,0
    for i in range(len(labels)):
      #if (labels[i] == 0 or labels[i] == 3): #T or Y
      if (prob_array[i][0][0] > tmp):
        tmp = prob_array[i][0][0] #T
        index = i
    if (index != -1):
      print(f"Likeliest candidate is at index {index} with probability {tmp}")
      return index
    else:
      raise Exception("No Likely T was found in the data")
  else:
    raise Exception("Array of not (X, 1, 4) shape")
  #O(n) preciso

def ttt_out(array, trueish_t):
    #set the most likely image to be a t to the actual t's. All else to y
    array= np.asarray(array) #check for array to be of numpy.ndarray
    if (array.ndim == 1):
      for a in range(0,len(array)):
        if (a==trueish_t):
          array[a] = 0 #the most likely T
        else:
          if (array[a] == 0): #if all other non-likely Ts
            array[a]=3 #Y
      return array
    else:
        raise Exception("Non 1-Dimensional array")

#Funzioni inizialmente scritte per scikit-learn, che supportavano la funzione predict_proba,
#che riportava array di probabilità, la quale non è presente su keras. ne creo una manuale

def get_prob_array(model, data):
  prob_array=[]
  prov_array=[]
  for one_case in range(0,len(data)):
    predictions_single = model.predict(data[one_case:(one_case + 1)])
    prob_array.append(predictions_single)
    prov_array.append(predictions_single.argmax())
  return prob_array, prov_array

#Impacchetto le due funzioni in un'unica chiamata
#Questa funzione prenderà in input il solo array delle Y e T

def cleandata(data, probabilities, labels):
  return ttt_out(data, close_enough_linear(probabilities, labels))
    
def predict():
	#carico le celle della griglia sul modello

	dir=sys.path[0]
	path="/output/"
	i,width,height=0,0,0
	data=[]

	!cd output/ && ls > path1

	#import image in dir
	with open((dir + path + "path1")) as file:
		for line in file.readlines():
		  keyword=line.rstrip()
		  if (keyword != "path1" and
		      keyword != "file.jpg" and
		      keyword != "finalimg.jpg" and
		      keyword != "photo.jpg"):
		    i+=1
		    filename=dir+path+keyword
		    if(filename[(len(filename)-6):(len(filename)-4)] == "10"):
		      #first row
		      width = i-1
		    print(filename)
		    imaget = cv.imread(filename, flags= cv.IMREAD_GRAYSCALE)
		    imaget = cv.bitwise_not(imaget)
		    imaget= imaget.reshape(28,28)
		    cv2_imshow(imaget)

		    #Normalizzo le immagini
		    #shape image as single array
		    imaget=imaget.reshape(1, 784)
		    imaget= imaget.astype("float32")/255
		    data.append(imaget)

	print(filename[(len(filename)- 6):(len(filename)- 4)])

	height = i//width

	print(f"width: {width}\nheight: {height:0.0f}")
	
	probabilities, labels = get_prob_array(bgty_model, data)

	for k in range(len(labels)):
		print(f"{probabilities[k]}")
	print(np.asarray(probabilities).shape)
	print(np.asarray(probabilities).ndim)
	print(labels)
	#passo le previsioni, dati e modello per ricavare il mio output
	#finale

	cleanup = cleandata(labels, probabilities, labels)

	print(f"l'array finale è {cleanup}")
	
	cleanup = [cleanup[a] for a in range(0,len(cleanup))]
	cleanup += [cleanup.index(0)]
	cleanup
