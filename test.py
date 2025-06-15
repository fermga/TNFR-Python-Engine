import networkx as nx
import matplotlib.pyplot as plt
import random
import hashlib
import imageio
import os
from tnfr.matrix.operators import aplicar_glifo, glifo_por_estructura
from tnfr.resonance.dynamics import inicializar_coordinador_temporal_canonico, integrar_tiempo_topologico_en_simulacion

# --- Función para convertir palabra en firma estructural ---
def firma_estructural(palabra):
    h = hashlib.md5(palabra.encode()).hexdigest()
    vf = (int(h[0:2], 16) % 100) / 100 + 0.5       # 0.5 a 1.5
    theta = (int(h[2:4], 16) % 100) / 100          # 0.0 a 1.0
    si = (int(h[4:6], 16) % 100) / 100             # 0.0 a 1.0
    return vf, theta, si

# --- Ingreso del estímulo externo ---
palabra = input("Ingrese una palabra para activar la red: ")
vf_est, theta_est, si_est = firma_estructural(palabra)
print(f"\nEstímulo estructural: νf={vf_est:.2f}, θ={theta_est:.2f}, Si={si_est:.2f}\n")

# Inicializar red
G = nx.Graph()

# Crear 20 nodos con valores aleatorios
for i in range(1, 21):
    G.add_node(f"N{i}",
        EPI=random.uniform(0.8, 1.2),
        νf=random.uniform(0.7, 1.2),
        ΔNFR=random.uniform(0.05, 0.2),
        Si=random.uniform(0.4, 0.6),
        θ=random.uniform(0.1, 0.9),
        glifo="AL",
        estado="activo"
    )

# Inicializar históricos necesarios
for n in G.nodes:
    G.nodes[n]["EPI_prev"] = G.nodes[n]["EPI"]
    G.nodes[n]["EPI_prev2"] = G.nodes[n]["EPI"]
    G.nodes[n]["θ_prev"] = G.nodes[n]["θ"]

# Conectar nodos si sus fases y sentidos son compatibles
nodos = list(G.nodes)
for i in range(len(nodos)):
    for j in range(i + 1, len(nodos)):
        ni, nj = G.nodes[nodos[i]], G.nodes[nodos[j]]
        if abs(ni["θ"] - nj["θ"]) < 0.2 and abs(ni["Si"] - nj["Si"]) < 0.2:
            G.add_edge(nodos[i], nodos[j])

# Historial
historial_glifos_por_nodo = {n: [] for n in G.nodes}
coordinador = inicializar_coordinador_temporal_canonico()

# Paleta de colores por glifo
colores_glifo = {
    "AL": "orange", "EN": "blue", "IL": "green", "OZ": "red", "UM": "purple",
    "RA": "cyan", "SHA": "gray", "VAL": "yellow", "NUL": "brown", "THOL": "pink",
    "ZHIR": "black", "NAV": "lime", "REMESH": "gold"
}

# Crear carpeta para imágenes
carpeta_frames = "frames_resonancia"
os.makedirs(carpeta_frames, exist_ok=True)

# Simular y visualizar
for paso in range(15):
    print(f"\nPaso {paso}")
    integrar_tiempo_topologico_en_simulacion(G, paso, coordinador)

    for n in G.nodes:
        nodo = G.nodes[n]

        # Comparar con estímulo
        delta_vf = abs(nodo["νf"] - vf_est)
        delta_theta = abs(nodo["θ"] - theta_est)
        delta_si = abs(nodo["Si"] - si_est)

        if delta_vf < 0.3 and delta_theta < 0.3 and delta_si < 0.3:
            glifo = "RA"  # resonancia
        else:
            glifo = glifo_por_estructura(nodo, G) or "AL"

        aplicar_glifo(G, nodo, n, glifo, historial_glifos_por_nodo, paso)

        print(f"{n} | EPI={nodo['EPI']:.2f} | νf={nodo['νf']:.2f} | ΔNFR={nodo['ΔNFR']:.2f} | Si={nodo['Si']:.2f} | Glifo={nodo['glifo']}")

    # Visualización
    colores = [colores_glifo.get(G.nodes[n]["glifo"], "white") for n in G.nodes]
    tamanos = [300 + G.nodes[n]["Si"] * 300 for n in G.nodes]
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color=colores, node_size=tamanos)
    plt.title(f"Paso {paso} | Entrada: {palabra}")
    filepath = os.path.join(carpeta_frames, f"frame_{paso:02d}.png")
    plt.savefig(filepath)
    plt.close()

# Crear GIF más lento
imagenes = [imageio.v2.imread(os.path.join(carpeta_frames, f)) for f in sorted(os.listdir(carpeta_frames)) if f.endswith(".png")]
imageio.mimsave("resonancia.gif", imagenes, duration=1.5)
print("\nGIF generado: resonancia.gif")


