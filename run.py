from script import config as config
from script.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# parser argumenata i definisanje
ap = argparse.ArgumentParser()

# --input: putanja do videa
ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")

# --output: putanja do mesta gde želimo da sačuvamo output video
ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")

# --display: prikaz aplikacije na ekranu dok se procesuira svaki frame. Vrednost 0 za background proces
ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")
args = vars(ap.parse_args())

# učitaj COCO labele klasa na kojima je naš YOLO model treniran
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# učitaj putanje do YOLO weights i konfiguracije modela
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# učitaj YOLO detektor objekata istreniran na COCO dataset-u u memoriju
print("[INFO] Loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# odredi imena potrebnih slojeva koje ćemo koristiti za procesiranje naših rezultata
layerNames = net.getLayerNames()
ln = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# inicijalizacija video stream-a (video ili kamera) i podešavanje da li se output upisuje u output video
print("[INFO] Accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

# iteracija kroz frejmove videa
while True:
    # učitaj sledeći frejm
    (grabbed, frame) = vs.read()

    # ako nema više frejmova došli smo do kraja stream-a
    if not grabbed:
        break

    # smanji frejm jer video može biti dosta veliki
    frame = imutils.resize(frame, width=700)

    # detektuj samo ljude na frejmu
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))

    # inicijalizacija setova koji će da nam predstavljaju
    violation = set()
    risky = set()
    safe = set()

    # bar dve osobe su detektovane iz razloga da bi mogli da izračnamo distancu između
    if len(results) >= 2:

        # uzimamo centre iz rezultata i izračunavamo Euklidsku distancu između svih parova centara
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # iteriramo kroz gornje trouglove matrice distance (simetrična matrica)
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # proveri distancu između centara da li je manja od minimalno konfigurisanog broja piksela
                if D[i, j] < config.MIN_DISTANCE:
                    # update našeg violation set-a sa indeksima para centara. Ako su preblizu dodamo u set.
                    violation.add(i)
                    violation.add(j)
                # proveri distancu između centara da li je manja od minimalno bezbedne razdaljine a da nije violation
                if (D[i, j] < config.MIN_SAFE_DISTANCE) and not violation:
                    risky.add(i)
                    risky.add(j)
                # proveri distancu između centara da li je veća od minimalne bezbedne razdaljine
                if D[i, j] > config.MIN_SAFE_DISTANCE and not violation and not risky:
                    safe.add(i)
                    safe.add(j)

    # prođi kroz rezultate
    for (i, (prob, bbox, centroid)) in enumerate(results):

        # uzimamo bounding box i koordinate centara, inicijalizujemo boju kojom ćemo prikazivati stanja
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # ako indeks par postoji u setu, dodeli mu određenu boju
        if i in violation:
            color = (0, 0, 255)
        if i in risky:
            color = (0, 255, 255)

        # iscrtaj (1) bounding box oko osobe i (2) centar te osobe
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 2)

    # ispis koliko ljudi krši pravila, koliko je u rizičnoj zoni i koliko je bezbedno
    violation_text = "Violation: {}".format(len(violation))
    cv2.putText(frame, violation_text, (10, frame.shape[0] - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

    risky_text = "Risky: {}".format(len(risky))
    cv2.putText(frame, risky_text, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

    safe_text = "Safe: {}".format(len(safe))
    cv2.putText(frame, safe_text, (10, frame.shape[0] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)

    # provera da li output frejm tj video treba biti prikazan na ekranu
    if args["display"] > 0:
        # prikaži output frejm
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # na `q` iskačemo iz petlje i prekidamo
        if key == ord("q"):
            break

    # ako imamo putanju do mesta gde čuvamo output video i video writer nije incijalizovan onda ga inicijalizuj
    if args["output"] != "" and writer is None:
        # inicijalizacija video writer-a
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # ako je video writer inicijalizovan, upiši frejm u video output file
    if writer is not None:
        writer.write(frame)