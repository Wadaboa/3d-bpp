# TODO

-   [ ] Refactor super-items
    -   [x] Check usefulness of horizontal super-items
    -   [x] Remove items from pool if they are part of an horizontal super-item
-   [ ] Add feasibility check (pure python/constraint programming/machine learning) before cp placement
-   [x] Reorder layer after final iteration by density
-   [x] (?) Revise s-shaped: Check case with more than 1 layer in new bin

## Nice to have

-   [ ] Add management of mass and density
-   [ ] Replace items remove in the "select_layers" phase with other not-placed items, maybe use maxrects
    -   [x] Temporary solution: "rearrange" with maxrects and re-iteration of all process with only not placed items
-   [ ] Add management of `Vertical Support` between items -> add spacing between as described in the paper
-   [ ] Add support for item rotation
