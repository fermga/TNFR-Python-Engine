import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

plt.style.use('seaborn-v0_8-darkgrid')

CONFIG_VISUAL = {
    'color_epi': '#1f77b4',
    'color_si': '#ff7f0e',
    'color_vf': '#2ca02c',
    'color_dNFR': '#d62728',
    'color_coherencia': '#9467bd',
    'figsize': (12, 6),
    'dpi': 300,
    'paleta': 'viridis',
    'estilo_lineas': {
        'linewidth': 1.5,
        'alpha': 0.7
    }
}

def graficar_dinamica_nodal_unificada(historia_epi, historia_si):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Subplot 1: Evolución de EPI
    for paso_data in historia_epi:
        for item in paso_data:
            ax1.scatter(item['paso'], item['EPI'], s=5, alpha=0.3, c='navy')
    ax1.set_ylabel('EPI', color='navy')
    ax1.tick_params(axis='y', labelcolor='navy')
    ax1.grid(True, alpha=0.1)
    
    # Subplot 2: Evolución de Si
    for paso_data in historia_si:
        for item in paso_data:
            ax2.scatter(item['paso'], item['Si'], s=5, alpha=0.3, c='crimson')
    ax2.set_xlabel('Paso de Simulación')
    ax2.set_ylabel('Si', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    ax2.grid(True, alpha=0.1)
    
    plt.suptitle('Dinámica Nodal Unificada')
    plt.tight_layout()
    plt.savefig("3_dinamica_nodal_unificada.png", dpi=300, bbox_inches='tight')
    plt.close()

def graficar_propiedades_estructurales(G):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Subplot 1: Distribución de νf
    vf_values = [n[1]['νf'] for n in G.nodes(data=True)]
    ax1.hist(vf_values, bins=30, color='teal', alpha=0.7)
    ax1.set_title('Distribución de Frecuencias Estructurales (νf)')
    ax1.set_xlabel('νf')
    ax1.set_ylabel('Frecuencia')
    
    # Subplot 2: Distribución de ΔNFR
    dNFR_values = [n[1]['ΔNFR'] for n in G.nodes(data=True)]
    ax2.hist(dNFR_values, bins=30, color='purple', alpha=0.7)
    ax2.set_title('Distribución de Gradientes Nodales (ΔNFR)')
    ax2.set_xlabel('ΔNFR')
    
    plt.tight_layout()
    plt.savefig("4_propiedades_estructurales.png", dpi=300)
    plt.close()

def graficar_topologia_temporal(historial_temporal):
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, :])
    
    # Línea principal: Coherencia Temporal
    ax1.plot([h['paso'] for h in historial_temporal], 
             [h['coherencia_temporal'] for h in historial_temporal], 
             lw=2, color='darkgreen')
    ax1.set_title('Evolución de la Coherencia Temporal Global')
    
    # Heatmap de Eventos Simultáneos
    eventos = [h['eventos_simultaneos'] for h in historial_temporal]
    ax2.imshow([eventos], cmap='YlOrRd', aspect='auto')
    ax2.set_title('Densidad de Eventos Simultáneos')
    
    # Diagrama de Caja de Grupos Resonantes
    grupos = [h['grupos_resonantes'] for h in historial_temporal]
    ax3.boxplot(grupos, vert=False)
    ax3.set_title('Distribución de Grupos Resonantes')
    
    # Análisis Multiresolución
    pasos = [h['paso'] for h in historial_temporal]
    coherencia = [h['coherencia_temporal'] for h in historial_temporal]
    ax4.plot(pasos, np.convolve(coherencia, np.ones(10)/10, mode='same'), 
            label='Tendencia (ventana=10)')
    ax4.scatter(pasos, coherencia, s=2, c='black', alpha=0.3, label='Muestras')
    ax4.legend()
    ax4.set_title('Análisis Multiresolución de Coherencia')
    
    plt.tight_layout()
    plt.savefig("5_topologia_temporal.png", dpi=300)
    plt.close()

def graficar_frecuencia_y_gradiente(G, nodos_seleccionados=None):
    if nodos_seleccionados is None:
        nodos_seleccionados = [
            n for n, d in G.nodes(data=True)
            if any(
                g in [gl for _, gl in d.get("historial_glifos", [])]
                for g in ["NAV", "IL", "ZHIR", "THOL", "RA"]
            )
        ][:10]

    def filtrar_historial(historial, umbral_max=1e6):
        pasos, valores = [], []
        for x in historial:
            if (
                isinstance(x[0], (int, float)) and
                isinstance(x[1], (int, float)) and
                np.isfinite(x[0]) and
                np.isfinite(x[1]) and
                abs(x[1]) < umbral_max
            ):
                pasos.append(x[0])
                valores.append(x[1])
        return pasos, valores

    def graficar(historiales, titulo, ylabel, filename):
        plt.figure(figsize=(12, 5))
        datos_validos = False
        for nodo_id in nodos_seleccionados:
            historial = G.nodes[nodo_id].get(historiales, [])
            pasos, valores = filtrar_historial(historial)
            if pasos and valores:
                plt.plot(pasos, valores, label=f"{nodo_id}")
                datos_validos = True
        if datos_validos:
            plt.title(titulo)
            plt.xlabel("Paso")
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid(True)
            try:
                plt.tight_layout()
            except Exception as e:
                print(f"tight_layout falló en {ylabel}: {e}")
            try:
                plt.savefig(filename, dpi=300)
            except Exception as e:
                print(f"savefig falló: {e}")
            plt.show()
        else:
            print(f"No hay datos válidos para {ylabel}")
 
def graficar_coherencia_total(historial_temporal):
    # Extraer datos de la estructura de diccionarios
    pasos = [x['paso'] for x in historial_temporal]
    valores = [x['coherencia_temporal'] for x in historial_temporal]
    
    plt.figure(figsize=(10, 5))
    plt.plot(pasos, valores, color="darkgreen", linewidth=2)
    plt.title("Coherencia total de la red C(t)")
    plt.xlabel("Paso")
    plt.ylabel("C(t) = coherencia temporal")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("2_coherencia_total.png", dpi=300)
    plt.close()  # Cambiado de plt.show() a plt.close() para consistencia

def generar_frames_optimizados(historia_G):
    config = {
        'dpi': 250,
        'node_size_scale': 0.8,
        'edge_alpha': 0.15,
        'paleta': plt.cm.viridis,
        'layout': nx.kamada_kawai_layout,
        'frame_size': (1400, 1400)
    }
    
    for i, G in enumerate(tqdm(historia_G)):
        plt.figure(figsize=(12,12), dpi=config['dpi'])
        pos = config['layout'](G)
        
        # Visualización mejorada
        nx.draw_networkx_nodes(G, pos, 
                             node_size=[np.log(n[1]['EPI']+1)*300*config['node_size_scale'] 
                                      for n in G.nodes(data=True)],
                             node_color=[n[1]['νf'] for n in G.nodes(data=True)],
                             cmap=config['paleta'],
                             edgecolors='black',
                             linewidths=0.3)
        
        nx.draw_networkx_edges(G, pos, 
                             alpha=config['edge_alpha'],
                             width=1,
                             edge_color='#000000')
        
        plt.axis('off')
        plt.savefig(f"frames_temp/frame_{i:04d}.png", 
                   bbox_inches='tight', 
                   pad_inches=0)
        plt.close()

def crear_gif_python(directorio_frames="frames_temp", archivo_salida="evolucion_red.gif", duracion=100, loop=0):
    """
    Crea un GIF animado a partir de imágenes PNG en un directorio usando solo Python.
    
    Args:
        directorio_frames (str): Ruta al directorio con las imágenes.
        archivo_salida (str): Nombre del archivo GIF de salida.
        duracion (int): Duración de cada frame en milisegundos (opcional, default=100).
        loop (int): Número de veces que se repite el GIF (0 = infinito).
    """
    # Lista ordenada de archivos PNG
    archivos = sorted(
        [f for f in os.listdir(directorio_frames) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )
    frames = []
    for archivo in archivos:
        ruta = os.path.join(directorio_frames, archivo)
        imagen = Image.open(ruta)
        frames.append(imagen)
    
    if frames:
        # Guardar el GIF
        frames[0].save(
            archivo_salida,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=duracion,
            loop=loop
        )
        print(f"GIF creado correctamente: {archivo_salida}")
        return True
    else:
        print("No se encontraron imágenes PNG en el directorio especificado.")
        return False

def visualizar_macronodos(G, macronodes_info):
    fig = plt.figure(figsize=(16, 12))
    
    # Layout jerárquico
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Capa 1: Nodos base
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n in G if not n.startswith('E_')],
                          node_size=100,
                          node_color='skyblue',
                          alpha=0.3)
    
    # Capa 2: Macronodos
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=macronodes_info['nodos'],
                          node_size=300,
                          node_color='crimson',
                          alpha=0.7)
    
    # Conexiones críticas
    nx.draw_networkx_edges(G, pos, 
                          edgelist=macronodes_info['conexiones'],
                          edge_color='darkred',
                          width=1.5,
                          alpha=0.4)
    
    # Etiquetas estratégicas
    labels = {n: n.split('_')[1] for n in macronodes_info['nodos']}
    nx.draw_networkx_labels(G, pos, 
                           labels=labels,
                           font_size=8,
                           font_color='white')
    
    plt.title('Topología Jerárquica de Macronodos')
    plt.savefig("7_macronodos_3d.png", dpi=300)
    plt.close()

def visualizar_red(G, nodos_emitidos=None):
    if len(G.nodes) == 0:
        print("Grafo vacío, no se puede visualizar.")
        return

    pos = nx.spring_layout(G, seed=42)
    nodos_validos = []
    colores = []
    tamaños = []

    for n in G.nodes:
        nodo = G.nodes[n]
        try:
            epi_raw = nodo.get("EPI", 1.0)
            θ_raw = nodo.get("θ", 0.0)

            epi = float(epi_raw) if np.isfinite(epi_raw) and float(epi_raw) >= 0 else 1.0
            θ = float(θ_raw) if np.isfinite(θ_raw) else 0.0

            tamaño = np.sqrt(epi) * 120
            if not np.isfinite(tamaño) or tamaño <= 0:
                continue

            nodos_validos.append(n)
            tamaños.append(tamaño)
            colores.append(θ)
        except Exception as e:
            print(f"Nodo inválido {n}: {e}")
            continue

    if not nodos_validos:
        print("Ningún nodo válido para dibujar.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2, ax=ax)
    nodos = nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodos_validos,
        node_color=colores,
        node_size=tamaños,
        cmap=plt.cm.plasma,
        alpha=0.9,
        ax=ax
    )

    if nodos_emitidos:
        etiquetas = {n: n for n in nodos_emitidos if n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=etiquetas, font_size=8, ax=ax)

    plt.axis("off")
    plt.title("Red Fractal Resonante", fontsize=14)
    plt.savefig("1_red.png", dpi=300)
    plt.close()

