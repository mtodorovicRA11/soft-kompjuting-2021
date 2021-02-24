# soft-kompjuting-2021
Repozitorijum za potrebe projekta iz predmeta Soft Kompjuting

Socijalna distanca je jedan od najefektivnijih načina za prevenciju virusa COVID-19. 
Identifikacija i sprovođenje ove mere je jedan od najbitnijih faktora koje su preporučene od strane zdravstvenih organizacija, tako da je to i motivacija izrade ovog projekta u cilju suzbijanja virusa na otvorenim i zatvorenim površinama.

Za potrebe projekta su uzeti nasumični video zapisi ljudi dok šetaju sa sigurnosnih kamera. Materijali se nalaze na sledećem linku: https://drive.google.com/drive/folders/1TXWoATd2o2I8oGH_bEOELG8lrXmJH_4S?usp=sharing. 
Skup će biti proširen ukoliko bude potrebe. Output će biti video sa oznakama da li ljudi krše pravila socijalne distance kao i faktor rizika obojen prikladnim bojama.

Ideja je napraviti alat koji može u "realnom vremenu" da detektuje da li je ispoštovana socijalna distanca. 
Projektno rešenje može se primetiti na primer u tržnim centrima, gde možemo videti da li ima previše ljudi unutra, da li su neke tačke kritične i obezbeđuje da na taj način možemo blagovremeno reagovati.
Sa stanovišta zdravstva, ovo može doprineti sprečavanju širenja virusa, identifikacije "problematičnih" lokacija i slično.

Za implementaciju koristiće se Python i njegova biblioteka OpenCV.
Za detekciju ljudi u kadru ćemo koristiti YOLOv3 treniran na COCO dataset-u.
NMS (Non-maxima suppression) za smanjenje preklapanja bounding box-a oko detektovanih objekata.
Izračunava se distanca između detektovanih ljudi u svakom kadru.

Validacija će biti vršena empirijski.
