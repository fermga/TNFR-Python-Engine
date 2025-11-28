import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def get_quadratic_residues(n):
    """Returns the set of quadratic residues modulo n (excluding 0)."""
    residues = set()
    for i in range(1, n):
        residues.add((i * i) % n)
    return residues

def create_paley_like_graph(n):
    """Creates a Paley-like graph for arbitrary n."""
    residues = get_quadratic_residues(n)
    # For Paley graph, we need -1 to be a quadratic residue if we want undirected
    # If n = 1 mod 4, -1 is a residue.
    # If n is composite, we just use the definition: edge if x-y in residues.
    # Note: This might create a directed graph if -1 is not a residue.
    # We'll treat it as undirected for spectral analysis (A + A.T if needed, or just A).
    
    adj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            diff = (i - j) % n
            if diff in residues:
                adj[i, j] = 1
    return adj

def analyze_spectrum(n):
    adj = create_paley_like_graph(n)
    # Check symmetry
    is_symmetric = np.allclose(adj, adj.T)
    
    if is_symmetric:
        evals = np.linalg.eigvalsh(adj)
    else:
        evals = np.linalg.eigvals(adj)
        
    # Sort real parts
    evals = np.sort(np.real(evals))
    
    unique_evals = np.unique(np.round(evals, 4))
    
    print(f"--- n = {n} ---")
    print(f"Symmetric: {is_symmetric}")
    print(f"Number of distinct eigenvalues: {len(unique_evals)}")
    print(f"Eigenvalues: {unique_evals}")
    
    # Theoretical check for primes
    if is_prime(n) and n % 4 == 1:
        expected = np.sort([(-1 + np.sqrt(n))/2, (-1 - np.sqrt(n))/2])
        print(f"Theoretical (Prime): {np.round(expected, 4)}")
    
    return evals

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

if __name__ == "__main__":
    print("=== Pillar 1: Composite Obstruction Investigation ===")
    print("Comparing spectra of Paley graphs (Prime) vs Paley-like graphs (Composite)\n")
    
    # Case 1: Prime (Rigid)
    analyze_spectrum(13) # 1 mod 4
    analyze_spectrum(17)
    
    # Case 2: Composite n = p*q (Broken Rigidity)
    # 15 = 3 * 5. (3, 5 are not 1 mod 4, but let's see)
    # 21 = 3 * 7
    # 33 = 3 * 11
    # 35 = 5 * 7 (Both 1 mod 4? No, 5 is 1, 7 is 3)
    # Let's try product of primes = 1 mod 4 to give it the best chance
    # 5 * 13 = 65
    analyze_spectrum(15)
    analyze_spectrum(21)
    analyze_spectrum(65) 
    
    # Case 3: Prime Power (Rigid?)
    # 9 = 3^2 (3 is 3 mod 4, so -1 not residue)
    # 25 = 5^2 (5 is 1 mod 4)
    analyze_spectrum(25)
