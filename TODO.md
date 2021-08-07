# TODO

- [ ] Riaggiungere gestione superitems
- [ ] Aggiunta feasibility check (procedurale/constraint programming/machine learning) prima del placement cp
- [x] Riordinamento finale layer per densità
- [ ] (?) Rivedere s-shaped: rivedere caso piu di un layer in nuovo bin

# Nice to have
- [ ] Aggiungere gestione peso e densità
- [ ] Sostituire item rimossi in fase di "select_layers" con altri item non piazzati, usando maxrects
  - [x] Soluzione temporanea: "rearrange" con maxrects e re-iterazione di tutto il processo con gli item non piazzati
- [ ] Supporto prodotti: gestione spaziatura come descritto nel paper
- [ ] Gestione rotazione
