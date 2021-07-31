# Todo

## Problemi in Input al modello

- [ ] Correggere gestione superitem in Heigth groups
  - Problema riguardo stesso Item nello stesso Layer dovuto a superitem Horizontal nei 2 assi -> (Messo in pausa, per ora risolto usando solo Vertical e Single superitems)
    - Suddividere gruppi in piu gruppi
    - Solo fra superitem con > 1 item e superitem con = 1 item
- [x] Warm start con maxrect calcolato su gruppi altezza

## Problemi interni al modello

- [ ] Risultati inconsistenti, stesso ordine, stessa configurazione diversi risultati
- [x] Salvare solo layer con alpha > 0 o tutti o solo quelli dell'ultimo ciclo
- [x] Rivedere constraint vari modelli e provare dopo ottimizzazione anche pricing problem placement
- [x] Rivedere constraint baseline (nel modello CP non si possono scrivere i constraint con "Constraint" e "SetCoefficient")
- [ ] Aggiunta feasibility check (procedurale/constraint programming/machine learning) prima del placement cp
- [x] Evitare duplicazione aggiunta layer in ciclo Column Generation

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
- [ ] Riordinamento per densità alla fine di tutto

## Aggiunte eventuali

- [ ] Gestione rotazione
- [ ] Oggetti fragili -> Peso + attributi particolari

## TODO tomorrow

- Finire s-shaped
  - Rivedere caso piu di un layer in nuovo bin
- Convertire oggetti in immutabili
  - "If layer in self.layers" lento -> fare come in superitems?
