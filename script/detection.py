from .config import NMS_THRESH, MIN_CONF, PEOPLE_COUNTER
import numpy as np
import cv2


# parametri
# frame: jedan frejm kao ulaz videa ili sa web kamere
# net: istreniran YOLO model za detekciju objekata
# ln: YOLO CNN output layer names
# personIdx: index za osobe jer YOLO model može da detektuje više tipova objekata

def detect_people(frame, net, ln, personIdx=0):
    # uzimamo dimenzije frejma
    (H, W) = frame.shape[:2]

    # inicijalizacija liste rezultata čiji će objekti koja će da sadrži parametre
    # (1) verovatnoću da li je detektovan objekat osoba
    # (2) koordinate bounding box-a (pravougaonika) za detekciju osobe
    # (3) centar objekta/osobe
    results = []

    # konstruišemo blob iz input frejma zbog preprocesinga
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # prosleđujemo blob detektoru, dobijamo bounding boxes i dodeljene verovatnoće za predikciju
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # inicijalizacija listi detektovanih bounding boxes, centara i faktora sigurnosti
    boxes = []
    centroids = []
    confidences = []

    # prolazimo kroz sve slojeve output-a, procesing
    for output in layerOutputs:
        # prolazimo kroz svaku detekciju
        for detection in output:
            # izvlačimo class ID i sigurnost tj verovatnoću detektovanog objekta
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filtriranje da proverimo da li je detektovani objekat osoba i da li je ispunjen minimalni uslov
            if classID == personIdx and confidence > MIN_CONF:
                # skaliramo bounding box-eve prema dimenzijama slike/frejma
                box = detection[0:4] * np.array([W, H, W, H])

                # YOLO vraća koordinate centra (x, y) kao i širinu i visinu bounding box-eva
                (centerX, centerY, width, height) = box.astype("int")

                # uzimamo koordinate bb da bi izvukli gornje levo koordinate objekta
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update vrednosti bb, centra i sigurnosti/verovatnoće
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # primeni non-maxima suppression to neutrališemo "slabe", preklapajuće bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ako je uključen brojač ljudi, ispiši vrednost
    if PEOPLE_COUNTER:
        human_count = "People in frame: {}".format(len(idxs))
        cv2.putText(frame, human_count, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 0, 0), 2)

    # ukoliko postoji bar jedna detekcija izvrši
    if len(idxs) > 0:
        # prolazimo krož indekse koje smo zadržali
        for i in idxs.flatten():
            # izvlačimo koordinate bounding box-eva
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update rezultata sa vrednostima, (1) verovatnoća, (2) koordinate bounding box-a osobe, (3) centar osobe
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # vraćamo listu retultata
    return results
