import sys
import os
import json
import numpy as np
import random
import re

np.seterr(divide='raise', over='ignore')

VARIABLES = {f'v{i}': i for i in range(10)}
OPERATORS = {
    "~": "~", "&": "&", "|": "|", "^": "^", "<<": "<<", ">>": ">>",
    "+": "+", "-": "-", "*": "*", "/": "//", "%": "%"
}

def parse_formula(formula: str):
    for var in VARIABLES: formula = formula.replace(var, f'values[{VARIABLES[var]}]')
    for op, py_op in OPERATORS.items(): formula = formula.replace(op, f' {py_op} ')
    return formula

def generate_samples(formula: str, num_samples: int):
    used_vars = sorted(set(re.findall(r'v([0-9])', formula)), key=int)
    parsed_formula = parse_formula(formula)
    samples = {}
    for i in range(num_samples):
        while True:
            values = [np.uint32(random.randint(0, (1 << 32) - 1)) for _ in range(10)]
            try: output = eval(parsed_formula); break
            except FloatingPointError: continue
        samples[i] = {
            "inputs": {var: hex(values[int(var)]) for var in used_vars},
            "output": hex(output)
        }
    return samples

def main():

    if len(sys.argv) != 3:
        print("Usage: python3 sample.py <formulas.txt> <num_samples>")
        return
    
    formulas_file = sys.argv[1]
    num_samples = int(sys.argv[2])
    output_dir = "generated_benchmarks"
    
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.rmdir(output_dir)
    os.makedirs(output_dir)
    
    with open(formulas_file, "r") as f:
        formulas = [line.strip() for line in f if line.strip()]
    
    for i, formula in enumerate(formulas):
        samples = generate_samples(formula, num_samples)
        with open(os.path.join(output_dir, f"{i}.json"), "w") as f:
            json.dump(samples, f, indent=4)
    
    print(f"Benchmarks generated in {output_dir}/")

if __name__ == "__main__":
    main()