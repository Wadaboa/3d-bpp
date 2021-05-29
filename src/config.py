# Pallet EUR 1 / Container 1A
# Pallet height represents the height of its base
pallet_lenght, pallet_width, pallet_height = 800, 1200, 145 # mm
container_lenght, container_width, container_height = 2330, 12000, 2200 # mm
pallet_load, container_load = 2490, 26480 # kg

# Product dimension ranges
num_products = 1000
min_lenght, min_width, min_height = 50, 50, 50 # mm
min_weight = 2 # kg
max_product_height = container_height - pallet_height

RANDOM_SEED = 42
