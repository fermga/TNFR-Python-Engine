import cv2
import matplotlib.pyplot as plt
import numpy as np
import tnfr.core

# Variables globales
prev_frame = None
coherencia_historial = []
min_area = 500  # Área mínima para considerar una región

def detectar_regiones_movimiento(frame, prev_frame):
    if prev_frame is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, prev_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regiones = []
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            regiones.append((x, y, w, h))
    return regiones

def crear_nodos_tnfr(frame, regiones):
    datos = []
    for i, (x, y, w, h) in enumerate(regiones):
        roi = frame[y:y+h, x:x+w]
        brillo = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        color = np.mean(roi, axis=(0,1))
        datos.append({
            'id': f'region_{i}',
            'forma_base': f"R{int(color[2])}G{int(color[1])}B{int(color[0])}B{int(brillo)}",
            'x': x, 'y': y, 'w': w, 'h': h
        })
    return datos

cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title('Evolución de la coherencia estructural TNFR')
ax.set_xlabel('Tiempo (frames)')
ax.set_ylabel('Coherencia estructural media')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    regiones = detectar_regiones_movimiento(frame, prev_frame)
    datos = crear_nodos_tnfr(frame, regiones)
    red = tnfr.core.ontosim.crear_red_desde_datos(datos)

    coherencias = [data['Si'] for _, data in red.nodes(data=True) if 'Si' in data]
    coherencia_media = np.mean(coherencias) if coherencias else 0.0
    coherencia_historial.append(coherencia_media)

    # Dibuja las regiones detectadas
    for x, y, w, h in regiones:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    # Visualiza la coherencia estructural media
    cv2.putText(frame, f"Coherencia TNFR: {coherencia_media:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Visualiza los atributos TNFR de cada nodo
    for i, (nodo, datos_nodo) in enumerate(red.nodes(data=True)):
        y_text = 60 + i*30
        cv2.putText(frame, f"{nodo}: Si={datos_nodo.get('Si', 0):.2f}", (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    cv2.imshow('Biofeedback TNFR', frame)

    # Actualiza la gráfica de evolución temporal
    line.set_xdata(range(len(coherencia_historial)))
    line.set_ydata(coherencia_historial)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    prev_frame = frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close()
