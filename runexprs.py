import os

# for i in range(7):
#     os.system(f'python3 main.py --id CAR_SHIELDED_PAPER_{i} --symbolic --stickiness 0.1 --paths-to-sample 999')

for p in [0.1, 0.5]:
    for i in range(5):
        os.system(f'python3 main.py --id CAR_PAPER_{p}_{i} --symbolic --stickiness {p}')
        