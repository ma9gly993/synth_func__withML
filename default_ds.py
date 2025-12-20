import csv
import itertools

N_BITS = 12

def complex_boolean_function(bits):
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11 = bits
    
    term1 = (x0 and x1) ^ (x2 or x3)
    term2 = (x4 ^ x5) and (not x6)
    term3 = (x7 and x8) or (x9 ^ x10)
    term4 = x11 and (x0 or x1)
    
    result = (term1 and term2) ^ (term3 or term4)
    return 1 if result else 0

def generate_dataset(filename, boolean_function):
    print(f"Генерация {filename}")

    all_inputs = list(itertools.product([0, 1], repeat=N_BITS))

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        header = [f"x{i}" for i in range(N_BITS)] + ["y"]
        writer.writerow(header)

        for bits in all_inputs:
            y = boolean_function(bits)
            writer.writerow(list(bits) + [y])

    print(f"Готово {filename} | samples = {len(all_inputs)}")

if __name__ == "__main__":
    generate_dataset("data/complex_bool_12bits.csv", complex_boolean_function)
