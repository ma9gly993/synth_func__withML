import csv
import itertools

# =============================
# НАСТРОЙКИ
# =============================
N_BITS = 12         # число булевых переменных
DATASET_SIZE = None # None = полный перебор (2^N_BITS)

# =============================
# БУЛЕВЫ ФУНКЦИИ
# =============================

def exact_2_or_0(bits):
    """1 если сумма битов = 0 или 2"""
    s = sum(bits)
    return 1 if s == 0 or s == 2 else 0


def mod3_count(bits):
    """1 если сумма битов ≡ 0 (mod 3)"""
    return 1 if sum(bits) % 3 == 0 else 0


def approx_30_percent_ones(bits):
    """1 если примерно 30% битов равны 1 (для балансировки классов)"""
    n = len(bits)
    ones_count = sum(bits)
    
    lower_bound = int(0.25 * n)
    upper_bound = int(0.35 * n) + 1
    
    return 1 if lower_bound <= ones_count <= upper_bound else 0


# =============================
# ГЕНЕРАЦИЯ ДАТАСЕТА
# =============================

def generate_dataset(filename, boolean_function):
    print(f"📦 Генерация {filename}")

    all_inputs = list(itertools.product([0, 1], repeat=N_BITS))

    if DATASET_SIZE:
        all_inputs = all_inputs[:DATASET_SIZE]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"x{i}" for i in range(N_BITS)] + ["y"]
        writer.writerow(header)

        for bits in all_inputs:
            y = boolean_function(bits)
            writer.writerow(list(bits) + [y])

    print(f"✅ {filename} | samples = {len(all_inputs)}")


# =============================
# ЗАПУСК
# =============================

if __name__ == "__main__":
    generate_dataset("exact_2_or_0__12bits.csv", exact_2_or_0)
    generate_dataset("mod3_count__12bits.csv", mod3_count)
    generate_dataset("approx_30pct_ones__12bits.csv", approx_30_percent_ones)
