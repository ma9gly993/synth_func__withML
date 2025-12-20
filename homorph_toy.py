# toy_homomorphic.py

import random

class ToyProbabilisticHE:
    def __init__(self, n, k):
        self.n = n
        self.k = k

    def encrypt(self, x):
        r = random.randint(0, self.n // 2)
        return (x + self.k + 2*r) % self.n

    def decrypt(self, c):
        return (c - self.k) % 2

    def add(self, c1, c2):
        return (c1 + c2) % self.n



def exact_2_or_0(bits):
    s = sum(bits)
    return 1 if s == 0 or s == 2 else 0


import csv
import itertools

# =============================
# НАСТРОЙКИ
# =============================
N_BITS = 12
N = 1_000_003     # большой модуль - обычно еще больше
K = 731291        # секрет - тоже больше
OUTPUT_FILE = "he_exact2_fail.csv"

he = ToyProbabilisticHE(N, K)

# =============================
# ГЕНЕРАЦИЯ
# =============================
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)

    header = [f"c{i}" for i in range(N_BITS)] + ["y"]
    writer.writerow(header)

    for bits in itertools.product([0, 1], repeat=N_BITS):
        y = exact_2_or_0(bits)
        encrypted_bits = [he.encrypt(b) for b in bits]

        writer.writerow(encrypted_bits + [y])

print(f"✅ Dataset saved as {OUTPUT_FILE}")
