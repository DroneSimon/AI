import cv2
import numpy as np
import xml.etree.ElementTree as ET

entrenamiento_fuego = 'Entrenamiento_SVM/fuego-last-chance-2.xml'
entrenamiento_humo = 'Entrenamiento_SVM/humo-last-chance-2.xml'
#Los colores son:azul,verde,rojo
COLORES = [(255,0,0),(0,255,0),(0,0,255)]
global max_humo
max_humo= -20.0
global max_fuego
max_fuego = -20.0

def detectar_fuego(imagen):
    if imagen!= None:
        #imagen = cv2.resize(imagen,(640,480))
        reconocedor = cargar_entrenamiento_fuego(imagen)
        #con la lista dibuja los rectangulos en la imagen
        imagen_reconocida = dibujar(imagen,[reconocedor],[1])
    else:
        print 'La imagen no es valida'
        imagen_reconocida = None
    return imagen_reconocida,porcentaje_error(max_fuego)

def detectar_humo(imagen):
    if imagen != None:
        #imagen = cv2.resize(imagen,(640,480))
        reconocedor = reconocer_humo(imagen)
        #con la lista de reconocimientos dibuja en la imagen
        imagen_reconocida = dibujar(imagen,[reconocedor],[0])
    else:
        print 'La imagen no es valida'
        imagen_reconocida = None
    return imagen_reconocida,porcentaje_error(max_humo)

def detectar_fuego_humo(imagen):
    if imagen != None:
        #imagen = cv2.resize(imagen,(640,480))
        #obtiene una lista de los tres tipos de fuego reconocidos [fuego,humo_blanco,humo_negro]
        reconocedores = reconocer(imagen)
        #con la lista dibuja los rectangulos en la imagen
        imagen = dibujar(imagen,reconocedores,[0,1])
    else:
        print "La direccion de la imagen no es correcta"
    return imagen,porcentaje_error(max_fuego),porcentaje_error(max_humo)

def porcentaje_error(x):
    if x <= -5:
        res = 0
    elif x < 0.5:
        res = 10 + (30 * x)
    else:
        res = 30 + (35*x)
    return int(res)

def reconocer(imagen):
    reconocedores = []
    reconocedor = reconocer_humo(imagen)
    reconocedores.append(reconocedor)
    recon_fuego = cargar_entrenamiento_fuego(imagen)
    reconocedores.append(recon_fuego)
    return reconocedores

def reconocer_humo(imagen):
    reconocedores = []
    hog = cargar_entrenamiento_humo(entrenamiento_humo,(64,64))
    res, pesos = hog.detectMultiScale(imagen, hitThreshold=0.9, scale=1.5)
    if len(res) != 0:
        global max_humo
        i = 0
        for x, y, w, h in res:
            rect = [x,y,x+w,y+h]
            reconocedores.append(rect)
            maximo = max(pesos[i],max_humo)
            i += 1
        reconocedores = np.array(reconocedores)
        reconocedores = agrupar_rec_humo(reconocedores)
        max_humo = max(maximo,max_humo)
    return reconocedores

def agrupar_rec_humo(reconocedores):
    grupos, pesos = cv2.groupRectangles(np.array(reconocedores).tolist(),1,0.2)
    return grupos

def dibujar(imagen, lista_rec,indices):
    copia = imagen.copy()
    if len(lista_rec) > 0:
        i = 0
        for grupo in lista_rec:
            if len(grupo) > 0:
                for x,y,w,h in grupo:
                    cv2.rectangle(copia,(x,y),(w,h),COLORES[indices[i]],3)
            i += 1
    else:
        print 'Error en el reconocimiento'
    return copia

def cargar_entrenamiento_humo(dir_archivo, tam_ventana):
    tree = ET.parse(dir_archivo)
    root = tree.getroot()
    vectors = root[0].find('support_vectors')
    vectors = vectors[0].text
    SV = [np.float32(line) for line in vectors.split()]
    labels = root[0].find('decision_functions')[0].find('rho')
    SV.append(-np.float32(labels.text))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    hog.setSVMDetector(np.array(SV))
    return hog

def cargar_entrenamiento_fuego(imagen):
    rec_fuego = []
    clasificador = entrenamiento_fuego
    img = cv2.cvtColor(imagen.copy(), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img)
    modelo = cv2.SVM()
    modelo.load(clasificador)
    mat_puntos= mapear(v.astype(np.uint8),imagen)
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8),(8,8),9)
    global max_fuego
    for x,y,w,h in mat_puntos:
        subMat = imagen[y:h,x:w]
        subMat = cv2.resize(subMat,(64,64))
        descriptor = hog.compute(subMat)
        descriptor = np.concatenate(descriptor)
        res = modelo.predict(descriptor)
        max_fuego = max(res, max_fuego)
        #Los rangos para el fuego son: ...
        if res > 0.3:
            rect = [x,y,w,h]
            rec_fuego.append(rect)
    return rec_fuego

def mapear(mat_v, img):
    mat_puntos = []
    ret, img_binaria = cv2.threshold(mat_v, 225, 235,cv2.THRESH_BINARY)
    contornos, herederos = cv2.findContours(img_binaria,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    int = 0
    for c in contornos:
        momentos = cv2.moments(c)
        if(momentos['m00']<700 and momentos['m00']>50):
            int +=1
            x = []
            y = []
            for i in c:
                for j in i:
                    x.append(j[0])
                    y.append(j[1])
            max_x, min_x, max_y, min_y = np.argmax(x), np.argmin(x), np.argmax(y), np.argmin(y)
            mat_puntos.append([x[min_x],y[min_y],x[max_x],y[max_y]])
    return mat_puntos
