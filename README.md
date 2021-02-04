# soft-kompjuting-2021
Repozitorijum za potrebe projekta iz predmeta Soft Kompjuting

Socijalna distanca je jedan od najefektivnijih načina za prevenciju virusa COVID-19. 
Identifikacija i sprovođenje ove mere je jedan od najbitnijih faktora koje su preporučene od strane zdravstvenih organizacija, tako da je to i motivacija izrade ovog projekta u cilju suzbijanja virusa.

Skup podataka (gde se nalazi, da li vršite data scraping ili je dataset već agregiran, da li ručno labelirate podatke i slično) *
Za potrebe projekta su uzeti nasumični video zapisi ljudi dok šetaju sa sigurnosnih kamera. Skup će biti proširen ukoliko bude potrebe.

Ideja je napraviti alat koji može u "realnom vremenu" da detektuje da li je ispoštovana socijalna distanca. 
Projektno rešenje može se primetiti na primer u tržnim centrima da li ima previše ljudi unutra, da li su neke tačke kritične i obezbeđuje da na taj način možemo blagovremeno reagovati.
Sa stanovišta zdravstva, ovo može doprineti sprečavanju širenja virusa, identifikacije "problematičnih" lokacija i slično.

Za implementaciju koristiće se Python i njegova biblioteka OpenCV
Detekcija ljudi u frame-u ce biti YOLO algoritam za detekciju i .
NMS (Non-maxima suppression) za smanjenje preklapanja bounding box-a oko detektovanih objekata.
Izračunava se distanca između detektovanih ljudi u svakom kadru.
