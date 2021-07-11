# Todo

## Problemi in Input al modello

- [ ] Correggere gestione superitem in Heigth groups
  - Problema riguardo stesso Item nello stesso Layer dovuto a superitem Horizontal nei 2 assi -> (Messo in pausa, per ora risolto usando solo Vertical e Single superitems)
    - Suddividere gruppi in piu gruppi
    - Solo fra superitem con > 1 item e superitem con = 1 item
- [ ] Warm start con maxrect calcolato su gruppi altezza

## Problemi interni al modello

- [ ] Risultati inconsistenti, stesso ordine, stessa configurazione diversi risultati
- [ ] Salvare solo layer con alpha > 0 o tutti o solo quelli dell'ultimo ciclo
- [ ] Rivedere constraint vari modelli e provare dopo ottimizzazione anche pricing problem placement e baseline

## Problemi in Output al modello

- [ ] Aggiungere gestione Weigth a quella della Densità con qualche priorità e spiegazione annessa
- [ ] Spezzare select_layer in fase di rimozione e filtraggio / fase di costruzione layer rimanenti per integrazione in ciclo di column-generation
- [ ] Sostituire item rimossi in fase di select_layers con altri item non piazzati, usando maxrects
  - [x] per ora: rearrange con maxrects e re-iterazione di tutto il processo con gli item non piazzati
- [ ] Gestione spaziatura
  - Spaziatura intra-layer spiegata nel paper?
    1. 2 modelli di linear programming -> Spaziatura mediante modelli per width and depth dimensions
    2. Soluzione di spostamento monodirezionale iterativa dei blocchi di un layer (Da pensare per bene come alternativa a 1)
  - Spaziatura inter-layer in altezza
    1. Controllare spazi vuoti anche in altezza al termine di tutto usando una delle 2 tecnice sopradescritte

## Aggiunte eventuali

- [ ] Gestione rotazione
- [ ] Oggetti fragili -> Peso + attributi particolari

## TODO tomorrow

- Finire s-shaped
  - Rivedere caso piu di un layer in nuovo bin
- Convertire oggetti in immutabili
  - "If layer in self.layers" lento -> fare come in superitems?
- Solver e search strategy per column generation
  - Da provare: convertire SP PLACEMENT in constraint programming (come fatto con SP NO PLACEMENT)
  - Da provare: rimuovere SP quando si usa maxrects -> ordinare items in base ai duali e chiamare maxrects (packing effettuato su cosa ci entra nel layer)
- Spaziatura
- Dobbiamo passare a SP solo gli item con dual > 0?
- Dobbiamo evitare di processare layer con alpha != 1 (anche in SP no placement)?
- RMP lavora su tutti i layer, solo su quelli nuovi o su una parte?
