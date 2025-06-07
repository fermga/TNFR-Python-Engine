# TNFR Simulador Extendido - Versión Canonizadora
# Incluye ecuación nodal explícita y cálculo del índice de sentido (Si)

import networkx as nx
import random
import math
import json
import csv
import matplotlib.pyplot as plt
import pandas as pd

# Palabras simbólicas base
palabras = [
    "humanidad", "humano", "persona", "gente", "hombre", "mujer", "bebé", "niño", "niña", "adolescente",
    "adulto", "adulta", "anciano", "anciana", "don", "doña", "señor", "señora", "caballero", "dama",
    "individuo", "cuerpo", "pierna", "pie", "talón", "espinilla", "rodilla", "muslo", "cabeza", "cara",
    "boca", "labio", "diente", "ojo", "nariz", "barba", "bigote", "cabello", "oreja", "cerebro",
    "estómago", "brazo", "codo", "hombro", "uña", "mano", "muñeca", "palma", "dedo", "trasero",
    "abdomen", "hígado", "músculo", "cuello", "corazón", "mente", "alma", "espíritu", "pecho", "cintura",
    "cadera", "espalda", "sangre", "carne", "piel", "hueso", "resfriado", "gripe", "diarrea", "salud",
    "enfermedad", "familia", "amigo", "amiga", "conocido", "conocida", "colega", "pareja", "esposo", "esposa",
    "matrimonio", "amor", "padre", "madre", "hermano", "hermana", "hijo", "hija", "abuelo", "abuela",
    "bisabuelo", "bisabuela", "nieto", "nieta", "bisnieto", "bisnieta", "primo", "prima", "tío", "tía",
    "sobrino", "sobrina", "criatura", "especie", "ser", "vida", "nacimiento", "reproducción", "muerte", "naturaleza",
    "campo", "bosque", "selva", "jungla", "desierto", "costa", "playa", "río", "laguna", "lago",
    "mar", "océano", "cerro", "monte", "montaña", "luz", "energía", "animal", "perro", "gato",
    "vaca", "cerdo", "caballo", "yegua", "oveja", "mono", "ratón", "rata", "tigre", "conejo",
    "dragón", "ciervo", "rana", "león", "jirafa", "elefante", "pájaro", "gallina", "gorrión", "cuervo",
    "águila", "halcón", "pez", "camarón", "langosta", "sardina", "atún", "calamar", "pulpo", "insecto",
    "bicho", "mariposa", "polilla", "saltamontes", "araña", "mosca", "mosquito", "cucaracha", "caracol", "babosa",
    "lombriz", "marisco", "molusco", "lagarto", "serpiente", "cocodrilo", "alimento", "comida", "bebida", "vegetal",
    "planta", "pasto", "césped", "flor", "fruta", "semilla", "árbol", "hoja", "raíz", "tallo",
    "hongo", "ciruela", "pino", "bambú", "nuez", "almendra", "castaña", "arroz", "avena", "cebada",
    "trigo", "verdura", "patatas", "papas", "judías", "guisantes", "porotos", "rábano", "zanahoria", "manzana",
    "naranja", "plátano", "pera", "castaño", "durazno", "tomate", "sandía", "carne", "gaseosa", "tiempo",
    "calendario", "edad", "época", "era", "fecha", "instante", "momento", "segundo", "minuto", "hora",
    "día", "semana", "entre semana", "fin de semana", "mes", "año", "década", "siglo", "milenio", "ayer",
    "hoy", "mañana", "amanecer", "mediodía", "tarde", "anochecer", "noche", "lunes", "martes", "miércoles",
    "jueves", "viernes", "sábado", "domingo", "ambiente", "espacio", "entorno", "área", "superficie", "volumen",
    "región", "zona", "lado", "mundo", "planeta", "sol", "luna", "estrella", "galaxia", "universo",
    "clima", "despejado", "nublado", "lluvia", "nieve", "viento", "trueno", "rayo", "tormenta", "cielo",
    "este", "oeste", "sur", "norte", "derecha", "izquierda", "diagonal", "exterior", "interior", "calor",
    "agua", "hielo", "vapor", "fuego", "gas", "aire", "atmósfera", "tierra", "piso", "suelo",
    "metal", "metálico", "hierro", "oro", "plata", "plomo", "sal", "barro", "lodo", "peso",
    "metro", "milímetro", "centímetro", "kilómetro", "litro", "gramo", "kilo", "cantidad", "total", "medida",
    "sociedad", "comunidad", "reunión", "encuentro", "estructura", "administración", "organización", "asociación", "empresa", "equipo",
    "autoridad", "cargo", "campaña", "club", "comisión", "congreso", "consejo", "partido", "país", "nación",
    "gobierno", "estado", "provincia", "departamento", "municipio", "democracia", "dictadura", "política", "político", "presidente",
    "ministro", "director", "parlamentario", "congresista", "senador", "diputado", "representante", "gobernador", "intendente", "alcalde",
    "policía", "bomberos", "capital", "ciudad", "población", "pueblo", "villa", "obligación", "libertad", "derecho",
    "permiso", "prohibición", "constitución", "ley", "decreto", "norma", "economía", "consumo", "demanda", "compañía",
    "comercio", "mercado", "servicio", "producto", "producción", "transacción", "almacén", "hotel", "fábrica", "cuenta",
    "boleto", "entrada", "dinero", "billete", "vuelto", "cambio", "máquina expendedora", "precio", "tarifa", "valor",
    "escritorio", "silla", "mesa", "cama", "dormitorio", "habitación", "cuarto", "oficina", "panel", "puerta",
    "ventana", "entrada", "hogar", "casa", "apartamento", "departamento", "edificio", "construcción", "elevador", "ascensor",
    "escalera", "aparato", "cámara", "aguja", "clavo", "hilo", "cuerda", "cordel", "cordón", "bolsillo",
    "bolso", "bolsa", "paraguas", "parasol", "pantalla", "pomo", "llave", "arma", "escultura", "libro",
    "revista", "cuadro", "grabado", "electricidad", "corriente", "base", "pata", "conexión", "ropa", "prenda",
    "manga", "solapa", "cuello", "botón", "cremallera", "cierre", "cinturón", "zapato", "gafas", "pantalón",
    "camisa", "camiseta", "zapatilla", "cordones", "abrigo", "chaqueta", "calcetines", "bragas", "calzón", "calzoncillo",
    "sujetador", "sostén", "falda", "transporte", "tránsito", "tráfico", "vehículo", "tren", "ferrocarril", "subterráneo",
    "metro", "camino", "vía", "ruta", "calle", "carretera", "autopista", "avenida", "estación", "parada",
    "avión", "aeropuerto", "automóvil", "coche", "auto", "bus", "autobús", "ómnibus", "ambulancia", "número",
    "alfabeto", "símbolo", "punto", "coma", "raíz", "origen", "fuente", "papel", "carta", "comunicación",
    "expresión", "voz", "texto", "periodismo", "periódico", "diario", "diccionario", "documento", "informe", "noticia",
    "acción", "actividad", "actor", "actriz", "admirar", "admitir", "adoptar", "adorar", "advertir", "afectar",
    "afirmar", "agitar", "agotar", "agradecer", "aguantar", "ahorrar", "alcanzar", "alegría", "alejar", "alimentar",
    "aliviar", "almacén", "alquilar", "ambiente", "ampliar", "analizar", "anunciar", "apagar", "aparecer", "aplaudir",
    "apoyar", "aprender", "apresurar", "aprobar", "aprovechar", "arreglar", "asegurar", "asistir", "asustar", "atacar",
    "atender", "atraer", "aumentar", "autor", "avanzar", "averiguar", "avisar", "ayudar", "bailar", "bajar",
    "balancear", "bañar", "barrer", "besar", "bloquear", "borrar", "brillar", "brindar", "buscar", "calcular",
    "calentar", "calmar", "cambiar", "caminar", "cancelar", "cantar", "capturar", "cargar", "casar", "celebrar",
    "cenar", "cerrar", "charlar", "chocar", "citar", "clarificar", "colaborar", "colocar", "comenzar", "comer",
    "cometer", "compartir", "competir", "completar", "comprar", "comprobar", "comunicar", "conceder", "concentrar", "concluir",
    "conducir", "confesar", "confirmar", "conocer", "conquistar", "conseguir", "conservar", "considerar", "construir", "consultar",
    "contar", "contener", "contestar", "continuar", "contribuir", "controlar", "convencer", "convertir", "convocar", "correr",
    "cortar", "crear", "crecer", "cruzar", "cubrir", "cuidar", "cumplir", "curar", "dañar", "debatir",
    "decidir", "decorar", "dedicar", "defender", "definir", "dejar", "delatar", "demostrar", "denunciar", "depender",
    "desarrollar", "descansar", "descargar", "descubrir", "desear", "desempeñar", "despertar", "destruir", "detener", "determinar",
    "dibujar", "dirigir", "discutir", "disfrazar", "disfrutar", "disminuir", "disparar", "disponer", "distribuir", "divertir",
    "dividir", "doblar", "dominar", "donar", "dormir", "dudar", "educar", "efectuar", "elegir", "eliminar",
    "empezar", "emplear", "emprender", "enamorar", "encender", "encerrar", "encontrar", "enfocar", "enfrentar", "engañar",
    "enojar", "enseñar", "entender", "enterrar", "entrar", "entregar", "enviar", "equilibrar", "escapar", "esconder",
    "escribir", "escuchar", "esperar", "establecer", "estimar", "estudiar", "evitar", "examinar", "exigir", "existir",
    "explicar", "explorar", "expresar", "extender", "fabricar", "facilitar", "fallar", "fascinar", "felicitar", "fijar",
    "filmar", "firmar", "flotar", "formar", "fortalecer", "fotografiar", "freír", "fumar", "funcionar", "ganar",
    "gastar", "gobernar", "golpear", "grabar", "graduar", "gritar", "guardar", "guiar", "gustar", "habitar",
    "hablar", "hallar", "heredar", "herir", "hervir", "huir", "iluminar", "imaginar", "impedir", "implementar",
    "importar", "impresionar", "incluir", "indicar", "informar", "iniciar", "insistir", "inspirar", "instalar", "intentar",
    "interesar", "interrumpir", "introducir", "invadir", "inventar", "invertir", "investigar", "invitar", "jugar", "juntar",
    "justificar", "lanzar", "lavar", "leer", "levantar", "liberar", "limpiar", "limitar", "llegar", "llenar",
    "llevar", "llorar", "llover", "lograr", "luchar", "manejar", "mantener", "marcar", "marchar", "matar",
    "medir", "mejorar", "mentir", "merecer", "mirar", "modificar", "molestar", "montar", "morir", "mostrar",
    "mover", "nadar", "nacer", "necesitar", "negar", "notar", "obedecer", "observar", "obtener", "ocupar",
    "ofrecer", "olvidar", "operar", "opinar", "organizar", "pagar", "parar", "parecer", "partir", "pasar",
    "patinar", "pedir", "pegar", "pensar", "perder", "perdonar", "permitir", "perseguir", "persuadir", "pesar",
    "pintar", "planear", "plantar", "platicar", "poder", "poner", "practicar", "predecir", "preferir", "preparar",
    "presentar", "preservar", "prestar", "pretender", "prevenir", "probar", "producir", "prohibir", "prometer", "proteger",
    "proveer", "publicar", "quedar", "quejar", "quemar", "querer", "quitar", "realizar", "recibir", "reclamar",
    "recomendar", "reconocer", "recordar", "recorrer", "recrear", "reducir", "reflejar", "reforzar", "regalar", "regresar",
    "reír", "relajar", "relatar", "reparar", "repetir", "resaltar", "rescatar", "resistir", "resolver", "respetar",
    "responder", "restaurar", "resultar", "retirar", "reunir", "revelar", "rezar", "robar", "romper", "saber",
    "sacar", "saltar", "saludar", "salvar", "satisfacer", "secar", "seguir", "seleccionar", "sembrar", "sentar",
    "sentir", "separar", "servir", "significar", "simbolizar", "simpatizar", "simplificar", "sintetizar", "situar", "soñar",
    "soportar", "subir", "suceder", "sugerir", "superar", "suponer", "suspender", "sustituir", "tardar", "temer",
    "tender", "terminar", "tirar", "tocar", "tomar", "trabajar", "traducir", "traer", "tratar", "triunfar",
    "unir", "usar", "utilizar", "vaciar", "valorar", "vender", "venir", "ver", "viajar", "vigilar",
    "visitar", "vivir", "volar", "volver", "votar", "zanjar"
]

def agrupar_por_tema(palabras):
    categorias = {
        "cuerpo": ['abdomen', 'barba', 'bigote', 'boca', 'brazo', 'cabello', 'cabeza', 'cadera', 'cara', 'carne',
                   'cerebro', 'cintura', 'codo', 'corazón', 'cuello', 'cuerpo', 'dedo', 'diente', 'espalda', 'espinilla',
                   'estómago', 'hombro', 'hueso', 'hígado', 'labio', 'mano', 'muslo', 'muñeca', 'músculo', 'nariz',
                   'ojo', 'oreja', 'palma', 'pecho', 'pie', 'piel', 'pierna', 'rodilla', 'sangre', 'talón', 'trasero', 'uña'],
        
        "relaciones": ['abuela', 'abuelo', 'amiga', 'amigo', 'amor', 'bisabuela', 'bisabuelo', 'bisnieta', 'bisnieto',
                       'colega', 'conocida', 'conocido', 'esposa', 'esposo', 'familia', 'gente', 'hermana', 'hermano',
                       'hija', 'hijo', 'madre', 'matrimonio', 'nieta', 'nieto', 'padre', 'pareja', 'persona', 'prima',
                       'primo', 'sobrina', 'sobrino', 'tía', 'tío'],
        
        "identidad": ['adolescente', 'adulta', 'adulto', 'anciana', 'anciano', 'bebé', 'caballero', 'dama', 'don', 'doña',
                      'hombre', 'humanidad', 'humano', 'individuo', 'mujer', 'niña', 'niño', 'señor', 'señora'],
        
        "salud": ['diarrea', 'enfermedad', 'gripe', 'resfriado', 'salud'],

        "espiritualidad": ['alma', 'espíritu', 'ser'],

        "naturaleza": ['criatura', 'especie', 'vida', 'naturaleza', 'bosque', 'jungla', 'campo', 'río', 'montaña', 'océano',
                       'mar', 'playa', 'lago', 'laguna', 'selva', 'costa', 'desierto', 'cerro', 'monte', 'planeta',
                       'tierra', 'atmósfera', 'galaxia', 'universo'],

        "animales": ['animal', 'perro', 'gato', 'vaca', 'caballo', 'yegua', 'cerdo', 'oveja', 'mono', 'ratón', 'rata',
                     'conejo', 'ciervo', 'tigre', 'león', 'jirafa', 'elefante', 'pájaro', 'águila', 'cuervo', 'gorrión',
                     'gallina', 'halcón', 'pez', 'sardina', 'atún', 'camarón', 'pulpo', 'calamar', 'langosta', 'insecto',
                     'araña', 'mosca', 'mosquito', 'cucaracha', 'caracol', 'babosa', 'lombriz', 'molusco', 'bicho', 'dragón'],

        "plantas y alimentos": ['planta', 'vegetal', 'flor', 'árbol', 'semilla', 'fruta', 'nuez', 'almendra', 'castaña',
                                'hoja', 'raíz', 'tallo', 'pasto', 'césped', 'hongo', 'bambú', 'pino', 'ciruela',
                                'manzana', 'pera', 'naranja', 'plátano', 'durazno', 'sandía', 'tomate', 'zanahoria',
                                'rábano', 'judías', 'porotos', 'guisantes', 'verdura', 'patatas', 'papas', 'arroz',
                                'trigo', 'avena', 'cebada', 'comida', 'bebida', 'alimento'],

        "tiempo": ['tiempo', 'día', 'noche', 'mes', 'año', 'hora', 'minuto', 'segundo', 'instante', 'momento',
                   'hoy', 'ayer', 'mañana', 'semana', 'entre semana', 'fin de semana', 'década', 'siglo', 'milenio',
                   'amanecer', 'tarde', 'anochecer', 'mediodía', 'lunes', 'martes', 'miércoles', 'jueves', 'viernes',
                   'sábado', 'domingo', 'fecha', 'calendario', 'edad', 'época', 'era'],

        "espacio": ['ambiente', 'espacio', 'entorno', 'área', 'superficie', 'volumen', 'región', 'zona', 'lado', 'mundo',
                    'exterior', 'interior', 'diagonal', 'norte', 'sur', 'este', 'oeste', 'derecha', 'izquierda', 'cielo'],

        "elementos y clima": ['luz', 'energía', 'agua', 'fuego', 'aire', 'gas', 'hielo', 'vapor', 'clima', 'despejado',
                              'nublado', 'lluvia', 'nieve', 'viento', 'trueno', 'rayo', 'tormenta', 'calor'],

        "materia": ['metal', 'hierro', 'plomo', 'plata', 'oro', 'sal', 'barro', 'lodo', 'tierra', 'piso', 'suelo',
                    'peso', 'gramo', 'kilo', 'litro', 'metro', 'milímetro', 'centímetro', 'kilómetro', 'cantidad',
                    'total', 'medida'],

        "estructura social": ['sociedad', 'comunidad', 'reunión', 'encuentro', 'estructura', 'administración', 'organización',
                              'empresa', 'autoridad', 'club', 'campaña', 'asociación', 'equipo'],

        "gobierno y política": ['estado', 'gobierno', 'provincia', 'departamento', 'municipio', 'democracia', 'dictadura',
                                'política', 'político', 'presidente', 'ministro', 'director', 'parlamentario',
                                'congresista', 'senador', 'diputado', 'representante', 'gobernador', 'intendente', 'alcalde'],

        "ley y derecho": ['obligación', 'libertad', 'derecho', 'permiso', 'prohibición', 'constitución', 'ley', 'decreto', 'norma'],

        "economía": ['economía', 'dinero', 'producto', 'valor', 'precio', 'tarifa', 'mercado', 'compañía', 'comercio',
                     'servicio', 'producción', 'transacción', 'boleto', 'cuenta', 'billete', 'vuelto', 'cambio', 'almacén'],

        "vivienda y objetos": ['hogar', 'casa', 'apartamento', 'edificio', 'departamento', 'dormitorio', 'habitación',
                               'cuarto', 'escritorio', 'mesa', 'silla', 'puerta', 'ventana', 'ascensor', 'escalera',
                               'cama', 'construcción', 'panel', 'bolso', 'bolsa', 'bolsillo', 'pantalla', 'llave']
    }

    agrupadas = []
    usadas = set()

    for grupo in categorias.values():
        agrupadas.extend(grupo)
        usadas.update(grupo)

    otras = [p for p in palabras if p not in usadas]
    agrupadas.extend(otras)

    return agrupadas, {p: cat for cat, grupo in categorias.items() for p in grupo}

palabras, categorias_palabra = agrupar_por_tema(palabras)

# Glifos como transformadores estructurales
historial_glifos_por_nodo = {}

glifos = {
    "A'L": lambda n: n.update({"EPI": n["EPI"] + 0.3, "νf": n["νf"] + 0.1, "estado": "emisión"}),
    "E'N": lambda n: n.update({"νf": n["νf"] + 0.05, "ΔNFR": n["ΔNFR"] - 0.05, "estado": "recepción"}),
    "I'L": lambda n: n.update({"ΔNFR": max(0, n["ΔNFR"] - 0.2), "estado": "coherencia"}),
    "O'Z": lambda n: n.update({"ΔNFR": n["ΔNFR"] + 0.3, "estado": "disonancia"}),
    "U'M": lambda n: n.update({"νf": n["νf"] + 0.1, "θ": n["θ"] + 0.1, "estado": "acoplamiento"}),
    "R'A": lambda n: n.update({"EPI": n["EPI"] * 1.1, "estado": "resonancia"}),
    "SH'A": lambda n: n.update({"νf": 0, "estado": "silencio"}),
    "VA'L": lambda n: n.update({"EPI": n["EPI"] * 1.2, "estado": "expansión"}),
    "NU'L": lambda n: n.update({"EPI": n["EPI"] * 0.8, "estado": "contracción"}),
    "T'HOL": lambda n: n.update({"θ": n["θ"] + math.pi / 4, "estado": "autoorganización"}),
    "Z’HIR": lambda n: n.update({"θ": n["θ"] + random.uniform(-0.5, 0.5), "estado": "mutación"}),
    "NA’V": lambda n: n.update({"EPI": n["EPI"] + 0.2, "estado": "transición"}),
    "RE'MESH": lambda n: n.update({"EPI": n["EPI"] * 0.95 + 0.1, "estado": "recursividad"})
}

def crear_red(palabras):
    G = nx.Graph()
    for p in palabras:
        G.add_node(p,
                   νf=random.uniform(0.5, 1.5),
                   ΔNFR=random.uniform(-0.5, 0.5),
                   θ=random.uniform(0, 2 * math.pi),
                   EPI=random.uniform(0.0, 0.5),
                   EPI_prev=0.0,
                   EPI_prev2=0.0,
                   EPI_anterior=0.0,
                   Si=0.0,
                   estado="latente",
                   glifo=None,
                   categoria=categorias_palabra.get(p, "otras"))
    for i in range(len(palabras)):
        for j in range(i + 1, len(palabras)):
            if random.random() < 0.25:
                G.add_edge(palabras[i], palabras[j])
    return G

def ecuacion_nodal(nodo):
    dEPI = nodo["νf"] * nodo["ΔNFR"]
    nodo["EPI"] += dEPI
    return dEPI

def calcular_Si(nodo):
    fase = math.cos(nodo["θ"])
    Si = nodo["νf"] * (1 - abs(nodo["ΔNFR"])) * fase
    nodo["Si"] = round(Si, 4)
    return nodo["Si"]

def exportar_Si(G):
    data = [{"palabra": n, "Si": G.nodes[n]["Si"]} for n in G.nodes]
    df = pd.DataFrame(data)
    df.to_csv("indice_sentido.csv", index=False)

    # Graficar automáticamente
    df = df.sort_values(by="Si", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(df["palabra"], df["Si"], color="mediumorchid")
    plt.xticks(rotation=90)
    plt.title("Índice de Sentido (Si) por Nodo")
    plt.ylabel("Si")
    plt.xlabel("Palabra")
    plt.tight_layout()
    plt.savefig("indice_sentido_barras.png")
    plt.close()
    return df

def mostrar_Si_global(G):
    Si_total = [G.nodes[n]["Si"] for n in G.nodes]
    promedio_Si = sum(Si_total) / len(Si_total)
    print(f"Índice de sentido promedio de la red: {round(promedio_Si, 4)}")
    nodos_altos = [n for n in G.nodes if G.nodes[n]["Si"] > 0.5]
    print(f"Nodos con alto Si (>0.5): {len(nodos_altos)}")

def detectar_EPIs_compuestas(G, umbral_Si=0.4, umbral_theta=0.5):
    subestructuras = []
    visitados = set()

    for nodo in G.nodes:
        if nodo in visitados:
            continue

        nodo_data = G.nodes[nodo]
        if nodo_data["Si"] < umbral_Si:
            continue

        grupo = [nodo]
        visitados.add(nodo)

        vecinos = list(G.neighbors(nodo))
        for vecino in vecinos:
            if vecino in visitados:
                continue
            vecino_data = G.nodes[vecino]
            if vecino_data["Si"] > umbral_Si and abs(nodo_data["θ"] - vecino_data["θ"]) < umbral_theta:
                grupo.append(vecino)
                visitados.add(vecino)

        if len(grupo) >= 2:
            subestructuras.append(grupo)

    return subestructuras

def interpretar_sintaxis_glífica(historial_glifos_por_nodo):
    secuencias_validas = {
        "nacimiento": ["A'L", "E'N", "I'L"],
        "bifurcación": ["T'HOL", "Z’HIR", "I'L"],
        "colapso": ["O'Z", "NU'L", "SH'A"],
        "resonancia": ["U'M", "R'A", "RE'MESH"],
        "mutación": ["NA’V", "Z’HIR", "VA'L"]
    }

    interpretaciones = []

    for nodo, secuencia in historial_glifos_por_nodo.items():
        if len(secuencia) < 3:
            continue

        última_tripla = secuencia[-3:]

        for nombre, patrón in secuencias_validas.items():
            if última_tripla == patrón:
                interpretaciones.append({
                    "nodo": nodo,
                    "secuencia": última_tripla,
                    "lectura": nombre
                })

    return interpretaciones

def aplicar_glifo(nodo, nombre_glifo):
    if nombre_glifo in glifos:
        glifos[nombre_glifo](nodo)
        nodo["glifo"] = nombre_glifo
        historial_glifos_por_nodo.setdefault(nodo.get("palabra", "?"), []).append(nombre_glifo)

        if nombre_glifo != "RE'MESH":
            detectar_memoria_y_aplicar_REMesh(nodo)


def detectar_memoria_y_aplicar_REMesh(nodo):
    nombre = nodo.get("palabra", "?")
    historial = historial_glifos_por_nodo.get(nombre, [])

    # Si los últimos 3 glifos son iguales, activar RE’MESH real
    if len(historial) >= 3 and historial[-1] == historial[-2] == historial[-3]:
        aplicar_glifo(nodo, "RE'MESH")
        # Efecto de memoria: estabiliza ΔNFR, sube ligeramente νf
        nodo["ΔNFR"] *= 0.7
        nodo["νf"] += 0.02

def emergencia_nodal(nodo, media_vf, std_dNFR):
    return nodo["νf"] > media_vf and abs(nodo["ΔNFR"]) < std_dNFR

def glifo_por_estructura(nodo):
    if nodo["ΔNFR"] > 0.4:
        return "O'Z"
    elif nodo["ΔNFR"] < -0.3:
        return "E'N"
    elif nodo["EPI"] > 2.5:
        return "I'L"
    elif 0.9 < nodo["νf"] < 1.1 and abs(nodo["θ"] - math.pi) < 0.3:
        return "RE'MESH"
    elif nodo["νf"] > 1.2 and nodo["θ"] > 4:
        return "VA'L"
    elif nodo["νf"] > 1.2 and nodo["ΔNFR"] < 0:
        return "R'A"
    elif nodo["νf"] < 0.5:
        return "A'L"
    elif nodo["νf"] > 0.8 and nodo["ΔNFR"] < 0.2:
        return "U'M"
    elif 1.5 < nodo["EPI"] < 2:
        return "NU'L"
    elif 1 < nodo["θ"] < 2:
        return "T'HOL"
    elif abs(nodo["θ"] - 2 * math.pi) < 0.2 or abs(nodo["θ"]) < 0.2:
        return "Z’HIR"
    elif nodo["EPI"] < 0.2:
        return "SH'A"
    else:
        return "NA’V"

def acoplar_nodos(G, umbral_θ=0.5, umbral_dNFR=0.4):
    for u, v in G.edges():
        nodo_u = G.nodes[u]
        nodo_v = G.nodes[v]

        if abs(nodo_u["θ"] - nodo_v["θ"]) < umbral_θ and \
           abs(nodo_u["ΔNFR"]) < umbral_dNFR and \
           abs(nodo_v["ΔNFR"]) < umbral_dNFR:

            # Promediar fase
            θ_prom = (nodo_u["θ"] + nodo_v["θ"]) / 2
            nodo_u["θ"] = θ_prom
            nodo_v["θ"] = θ_prom

            aplicar_glifo(nodo_u, "U'M")
            aplicar_glifo(nodo_v, "U'M")

def evaluar_mutacion_ZHIR(nodo, ξ=0.3):
    delta_EPI = abs(nodo["EPI"] - nodo.get("EPI_anterior", 0))
    if delta_EPI > ξ:
        aplicar_glifo(nodo, "Z’HIR")

def evaluar_autoorganizacion_THOL(nodo, τ=0.25):
    a = abs(nodo["EPI"] - 2 * nodo["EPI_prev"] + nodo["EPI_prev2"])
    if a > τ:
        aplicar_glifo(nodo, "T’HOL")

def simular_emergencia(G, pasos=1000):
    historia_epi = []
    historia_glifos = ["paso,palabra,glifo"]
    for paso in range(pasos):
        paso_data = []
        todos_vf = [n["νf"] for _, n in G.nodes(data=True)]
        media_vf = sum(todos_vf) / len(todos_vf)
        std_dNFR = sum(abs(n["ΔNFR"]) for _, n in G.nodes(data=True)) / len(G)
     
        acoplar_nodos(G)

        for n in G.nodes:
            nodo = G.nodes[n]

            # Pausa real si el nodo está en silencio
            if nodo["estado"] == "silencio":
                if abs(nodo["ΔNFR"] - nodo["νf"]) < 0.05:
                    aplicar_glifo(nodo, "NA’V")  # posible transición coherente
                else:
                    continue  # pausa activa

          
            nodo["palabra"] = n  # Para referencia en salida
            if emergencia_nodal(nodo, media_vf, std_dNFR):
                glifo = glifo_por_estructura(nodo)
                aplicar_glifo(nodo, glifo)
                historia_glifos.append(f"{paso},{n},{glifo}")
            paso_data.append({"palabra": n, "EPI": round(nodo["EPI"], 2)})
            evaluar_mutacion_ZHIR(nodo)
            evaluar_autoorganizacion_THOL(nodo)

            # Actualizar historia EPI
            nodo["EPI_prev2"] = nodo["EPI_prev"]
            nodo["EPI_prev"] = nodo["EPI"]

        historia_epi.append(paso_data)
    with open("historial_glifos.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(historia_glifos))
    epi_compuestas = detectar_EPIs_compuestas(G)
    with open("epis_compuestas.json", "w", encoding="utf-8") as f:
        json.dump(epi_compuestas, f, indent=4, ensure_ascii=False)
    lecturas = interpretar_sintaxis_glífica(historial_glifos_por_nodo)
    with open("lecturas_glificas.json", "w", encoding="utf-8") as f:
        json.dump(lecturas, f, indent=4, ensure_ascii=False)

    return historia_epi, G, epi_compuestas, lecturas

# Ejecutar simulación
G = crear_red(palabras)
epi_historial, G, epi_compuestas, lecturas = simular_emergencia(G)

# Procesar nodos emitidos
emitidos = [n for n, d in G.nodes(data=True) if d["estado"] != "latente"]

# 🌀 TNFR: Emisión estructural según condiciones resonantes
umbral_epi = 2.5
emitidos_final = [
    n for n in emitidos
    if G.nodes[n]["EPI"] > umbral_epi and G.nodes[n]["Si"] > 0
]

if len(emitidos_final) == 0:
    emitidos_final = sorted(
        [n for n in emitidos if G.nodes[n]["estado"] != "latente"],
        key=lambda n: G.nodes[n]["EPI"], reverse=True
    )[:10]  # fallback operativo

resultado = [
    {"palabra": n, "glifo": G.nodes[n]["glifo"], "EPI": round(G.nodes[n]["EPI"], 2)}
    for n in emitidos_final
]

print("Palabras emitidas por coherencia estructural:")
for r in resultado:
    categoria = G.nodes[r['palabra']].get("categoria", "sin categoría")
    print(f"- {r['palabra']} → {r['glifo']} (EPI: {r['EPI']}) | Categoría: {categoria}")

# Guardar evolución de EPI
epi_csv_data = ["paso,palabra,EPI"]
for i, paso in enumerate(epi_historial):
    for nodo in paso:
        epi_csv_data.append(f"{i},{nodo['palabra']},{nodo['EPI']}")

with open("evolucion_epi.csv", "w", encoding="utf-8") as f:
    f.write("\n".join(epi_csv_data))

# Visualizar evolución de EPI
historial_df = pd.DataFrame([{"paso": i, "palabra": n["palabra"], "EPI": n["EPI"]} for i, paso in enumerate(epi_historial) for n in paso])

if emitidos_final:
    plt.figure(figsize=(12, 6))
    for palabra in emitidos_final:
        datos = historial_df[historial_df["palabra"] == palabra]
        plt.plot(datos["paso"], datos["EPI"], label=palabra)
    plt.title("Evolución del EPI de palabras emitidas")
    plt.xlabel("Paso")
    plt.ylabel("EPI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evolucion_epi_grafico.png")
    plt.close()
else:
    print("⚠️ No hubo palabras con emergencia nodal suficiente para graficar evolución EPI.")

# Diagnóstico estructural final
nodos_activos = [n for n, d in G.nodes(data=True) if d["estado"] != "latente"]
media_vf_total = sum([G.nodes[n]["νf"] for n in G.nodes]) / len(G)
promedio_dNFR = sum(abs(G.nodes[n]["ΔNFR"]) for n in G.nodes) / len(G)
conteo_estados = {}
for _, d in G.nodes(data=True):
    estado = d.get("estado", "indefinido")
    conteo_estados[estado] = conteo_estados.get(estado, 0) + 1

# 🧩 Diagnóstico simbólico por nodo
diagnostico = []

for nodo in G.nodes:
    nombre = nodo
    datos = G.nodes[nodo]
    glifos_nodo = historial_glifos_por_nodo.get(nombre, [])
    
    # Ver si mutó
    mutó = "Z’HIR" in glifos_nodo
    
    # Ver si está en EPI compuesta
    en_epi = any(nombre in grupo for grupo in epi_compuestas)
    
    # Interpretación sintáctica si la hay
    lectura = next((l["lectura"] for l in lecturas if l["nodo"] == nombre), None)
    
    diagnostico.append({
    "palabra": nombre,
    "glifos": glifos_nodo,
    "lectura_sintactica": lectura,
    "mutó": mutó,
    "en_epi_compuesta": en_epi,
    "Si": datos.get("Si", 0),
    "estado": datos.get("estado", "latente"),
    "categoría": datos.get("categoria", "sin categoría")
})

with open("diagnostico_simbolico.json", "w", encoding="utf-8") as f:
    json.dump(diagnostico, f, indent=4, ensure_ascii=False)


print("\n🔍 Diagnóstico estructural global:")
print(f"Nodos activos: {len(nodos_activos)} / {len(G)}")
print(f"νf medio de red: {round(media_vf_total, 3)} | ΔNFR medio: {round(promedio_dNFR, 3)}")
print("Distribución de estados nodales:")
# Calcular índice de sentido por nodo
for n in G.nodes:
    calcular_Si(G.nodes[n])

# Exportar e imprimir índice de sentido global
exportar_Si(G)
mostrar_Si_global(G)

for estado, count in conteo_estados.items():
    print(f"- {estado}: {count}")
