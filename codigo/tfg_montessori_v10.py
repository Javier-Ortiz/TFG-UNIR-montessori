######################################
# --- PROYECTO PARA TFG DE LA UNIR ---
# Autor: Francisco Javier Ortiz Gonzalez
# Fecha: Diciembre, 2025
# Licencia: AGPL_v3
######################################

import cv2
import time
import csv
from collections import defaultdict
from ultralytics import YOLO
import numpy as np 
import os 
import sys 

#####################################
# --- PARA LA GENERACIÓN DE RUTAS ---
#####################################
RUTA_COMPLETA_SCRIPT = os.path.abspath(__file__)    #.../TFG/codigo/archivo.py
SCRIPT_DIR = os.path.dirname(RUTA_COMPLETA_SCRIPT)  # subimos un nivel  .../TFG/codigo
RUTA_RAIZ_PROYECTO = os.path.dirname(SCRIPT_DIR)    # subimos un nivel  .../TFG

##################################
# --- PARAMETROS CONFIGURABLES ---
##################################

# ESTABLECER CAPTURA DE VIDEO
# Se indica 0 para webcam o la ruta al archivo de video
#SOURCE = 0 # Webcam
SOURCE = os.path.join(RUTA_RAIZ_PROYECTO, "videos","cama_elastica.mp4") 

# --- CONTROL DE COMPORTAMIENTO ---
PRINT_CONSOLA = True            # Si es True, muestra estadísticas en la consola en tiempo real.
PRINT_PANTALLA = True           # Si es True, muestra el video etiquetado en una ventana en tiempo real.
GENERAR_VIDEO_ETIQUETADO = True # Si es True, se genera el archivo de video etiquetado.
GUARDAR_HIL_ID = True           # Si es True, guarda recortes del bounding box de cada.

# Nombre de las zonas
ZONA_1 = "ZONA1"
ZONA_2 = "ZONA2"
ZONA_3 = "ZONA3"
ZONA_4 = "ZONA4"

NOMBRES_ZONAS = [ZONA_1, ZONA_2, ZONA_3, ZONA_4]


# Archivo donde se guardarán las estadísticas
OUTPUT_CSV_FILE = os.path.join(RUTA_RAIZ_PROYECTO, "estadisticas","estadisticas_permanencia.csv")

# Ruta donde se generará el video etiquetado
OUTPUT_VIDEO_FILE = os.path.join(RUTA_RAIZ_PROYECTO, "videos","output_video_etiquetado.mp4")

# Carpeta principal para donde se almacenaran los recortes de bounding box
OUTPUT_HIL_DIR = os.path.join(RUTA_RAIZ_PROYECTO, "HIL_ID") 

# Archivo de log de detecciones
OUTPUT_HIL_LOG = os.path.join(OUTPUT_HIL_DIR, "id_detection_log.csv") 

# Modelo de detección utilizado
MODELO_DETECCION = os.path.join(RUTA_RAIZ_PROYECTO,"yolo","yolov8s.pt")

# Archivo de configuración del Tracker utilizado para el seguimiento de objetos
TRACKER_CONFIG = os.path.join(RUTA_RAIZ_PROYECTO,"trackers","tracker_bytetrack_tfg_montessori_v1.yaml")  # Usado para hacer el refinamiento de parametros


# --- Comprobación de generación de rutas correctas ---
print("Ruta final del video (SOURCE):", SOURCE) 
print("Ruta final del archivo estadísticas (OUTPUT_CSV_FILE):", OUTPUT_CSV_FILE)
print("Ruta final del video etiquetado (OUTPUT_VIDEO_FILE):", OUTPUT_VIDEO_FILE) 
print("Ruta final del carpeta HIL_ID (OUTPUT_HIL_DIR):", OUTPUT_HIL_DIR)
print("Ruta final del log de capturas (OUTPUT_VIDEO_FILE):", OUTPUT_HIL_LOG) 
print("Ruta final del modelo YOLO (MODELO_DETECCION):", MODELO_DETECCION)
print("Ruta final del tracker (TRACKER_CONFIG):", TRACKER_CONFIG)

# Establecemos título de la ventana
TITULO_VENTANA = "Aplicacion de Seguimiento y Zonas"

# Clases de COCO que nos interesa detectar
CLASES_DE_INTERES = [0] # Con el índice 0 indicamos que sólo queremos que detecte PERSONAS


#################################################################################
# --- OPTIMIZACIÓN PROCESAMIENTO Vs Aumento de riesgo de errores en detección ---
#################################################################################

# Se establece cuántos frames deben ser saltados por cada frame procesado.
FRAMES_IGNORADOS = 0  # Por ejemplo, 4 significa procesar 1 de cada 5 frames (4 frames ignorados).

# Resolución de fotogramaS. Para mejorar el rendimiento de procesamiento
RESOLUCION_FOTOGRAMA = 480 

# Umbral confianza para considerar que es una persona
UMBRAL_CONFIANZA = 0.2

# variable para almacenar los FPS del video capturado
FPS_VIDEO = 0



###############################
# --- FUNCIONES AUXILIARES ---
###############################

def is_punto_en_zona(punto, poligono):
    """Verifica si un punto (centro inferior de la bbox) está dentro de un polígono (Zona de interés)."""
    return cv2.pointPolygonTest(poligono, (float(punto[0]), float(punto[1])), False) >= 0

def guardar_csv(tiempo_permanencia, NOMBRES_ZONAS, OUTPUT_CSV_FILE, totalFramesIgnorados):
    """Guarda las estadísticas generadas en tiempo real en un archivo CSV."""

    # Comprueba si existe la carpeta, en caso contrario la crea
    carpeta_estadisticas = os.path.dirname(OUTPUT_CSV_FILE)
    if not os.path.exists(carpeta_estadisticas):
        os.makedirs(carpeta_estadisticas)
        print(f"Directorio de estadísticas creado en: {carpeta_estadisticas}")
    with open(OUTPUT_CSV_FILE, 'w', newline='') as file:
        writer_csv = csv.writer(file)
        # Muestra por consola los frames ignorados
        print("\n--- Total de frames ignorados: {} ---".format(totalFramesIgnorados))
        
        headers = ["ID_PERSONA"] + NOMBRES_ZONAS
        writer_csv.writerow(headers)
        
        for p_id, zonas_data in tiempo_permanencia.items():
            row = [p_id] + [round(zonas_data.get(zona_name, 0), 2) for zona_name in NOMBRES_ZONAS]
            writer_csv.writerow(row)
    print("--- Estadísticas guardadas en: {} ---".format(OUTPUT_CSV_FILE))

def dibujar_estadisticas_consola(tiempo_permanencia, NOMBRES_ZONAS, fps_text, progreso_text):
    """Imprime el estado de las estadísticas y FPS en la consola."""
    print("--- ESTADÍSTICAS EN TIEMPO REAL - FPS({}) ---".format(fps_text))
    
    # Muestra el progreso, en el caso de estar capturando un video, en vez de webcam
    if progreso_text:
        print(progreso_text)
        print("-" * 42)
    
    # Encabezados de 4 zonas
    header = "{:<6}".format('ID')
    for name in NOMBRES_ZONAS:
        header += "{:<9}".format(name.split(' - ')[0])
    print(header)
    print("-" * 42)
    
    # Datos calculados
    for p_id, zonas in tiempo_permanencia.items():
        row_str = "P-{:<5}".format(p_id)
        for name in NOMBRES_ZONAS:
            tiempo = round(zonas[name], 1)
            tiempo_str = f"{tiempo}s"
            row_str += "{:<8}".format(tiempo_str)
        print(row_str)

def guardar_recorte_y_log(im0, bbox, track_id, idLog_contador, tiempo_permanencia, NOMBRES_ZONAS, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG):
    """Guarda el recorte de la persona en su subcarpeta ID y añade una línea al log de detecciones."""
    
    id_entero = track_id
    id_str = str(id_entero)
    
    # Crea subcarpeta para el ID si no existe aún
    ruta_subcarpeta = os.path.join(OUTPUT_HIL_DIR, "ID_{}".format(id_str))
    if not os.path.exists(ruta_subcarpeta):
        os.makedirs(ruta_subcarpeta)

    # Recorta la imagen para obtener los bbox (bbox en formato coordenadas xmin, ymin, xmax, ymax)
    x_min, y_min, x_max, y_max = bbox
    # Aseguraramos que las coordenadas no se salgan de la pantalla
    # cuando YOLO detecta objetos cerca de los límites intenta generar coordenadas fuera de pantalla
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(im0.shape[1], x_max)
    y_max = min(im0.shape[0], y_max)
    
    recorte = im0[y_min:y_max, x_min:x_max]   # coordenadas del recorte ajustado
    
    # Guarda el recorte
    nombre_archivo = "{:06d}.jpg".format(idLog_contador)  #formato XXXXXX.jpg
    ruta_archivo = os.path.join(ruta_subcarpeta, nombre_archivo)
    cv2.imwrite(ruta_archivo, recorte)
    
    # Genera línea de LOG
    # Obtenemos los tiempos de permanencia de la persona actual
    # Usamos el track_id (que es el ID de la persona) como clave para buscar los tiempos
    tiempos = [round(tiempo_permanencia[track_id].get(zona, 0), 2) for zona in NOMBRES_ZONAS]
    
    # idLog, idPersona, zona1, zona2, zona3, zona4
    datos_log = [idLog_contador, id_entero] + tiempos
    
    # Escribir en el archivo CSV de log principal
    with open(OUTPUT_HIL_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(datos_log)
    
    return idLog_contador + 1


#######################################
# --- INICIALIZACIÓN DE COMPONENTES ---
#######################################

def inicializar_sistema(SOURCE, MODELO_DETECCION, OUTPUT_VIDEO_FILE, NOMBRES_ZONAS, GENERAR_VIDEO_ETIQUETADO, GUARDAR_HIL_ID, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG, FRAMES_IGNORADOS):
    """
    Inicializa modelo, captura de video, escritor de video, y define zonas.
    Calcula el tiempo de muestreo corregido si existe ajuste para saltar frames
    """
    
    # Inicializar el modelo YOLO
    model = YOLO(MODELO_DETECCION)

    # Inicializar la captura de video
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise ValueError("Error: No se pudo abrir video {}".format(SOURCE))

    # Obtención de dimensiones y FPS
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # ancho del video
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))   # alto del video
    
    TOTAL_FRAMES = 0

    if SOURCE == 0:     # Comprueba si la captura es de una webcam
        FPS_VIDEO = 30.0   #FPS habitual en webcam
        print("FPS establecido a {:.2f}".format(FPS_VIDEO))
    else:   
        FPS_VIDEO = cap.get(cv2.CAP_PROP_FPS)       # FPS del video
        TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if FPS_VIDEO <= 0:
            raise ValueError("El archivo de video no detecta FPS.")
    
    # --- CÁLCULO DE TIEMPOS DE PERMANENCIA ---
    # Duración de un solo frame (TIEMPO_REAL_FRAME)
    TIEMPO_REAL_FRAME = 1 / FPS_VIDEO 
    
    # Debemos ajustar este valor si existe salto de frames por haber establecido frames ignorados
    # El frame procesado representa su propio tiempo más los frames ignorados
    TIEMPO_DE_MUESTREO_CORREGIDO = TIEMPO_REAL_FRAME * (1 + FRAMES_IGNORADOS)
    
    print("FPS del video: {:.2f}".format(FPS_VIDEO))
    print("Tiempo equivalente en (segundos) por cada frame procesado: {:.4f}".format(TIEMPO_DE_MUESTREO_CORREGIDO))
    # ---------------------------------------------
    
    # Si se ha establecido, se generará el video etiquetado
    writer = None
    if GENERAR_VIDEO_ETIQUETADO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Códec para MP4
        writer = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, FPS_VIDEO, (W, H)) 

    # Preparar carpeta y log de HIL_ID, si así ha sido establecido
    if GUARDAR_HIL_ID:
        if not os.path.exists(OUTPUT_HIL_DIR):
            os.makedirs(OUTPUT_HIL_DIR)
        
        # Crear encabezados del log principal
        headers = ["idLog", "idPersona"] + ["zona{}".format(i) for i in range(1, len(NOMBRES_ZONAS) + 1)]
        with open(OUTPUT_HIL_LOG, 'w', newline='') as f:
            writer_log = csv.writer(f)
            writer_log.writerow(headers)
        print("Carpeta principal y log de HIL_ID creados en: {}".format(OUTPUT_HIL_DIR))

    # Definición de ZONAS
    # Para las pruebas del prototipo se simplificará en una cuadricula de 2 X 2
    HALF_W = W // 2
    HALF_H = H // 2
    ZONAS = {} 
    ZONAS[NOMBRES_ZONAS[0]] = np.array([(0, 0), (HALF_W, 0), (HALF_W, HALF_H), (0, HALF_H)], dtype=np.int32)
    ZONAS[NOMBRES_ZONAS[1]] = np.array([(HALF_W, 0), (W, 0), (W, HALF_H), (HALF_W, HALF_H)], dtype=np.int32)
    ZONAS[NOMBRES_ZONAS[2]] = np.array([(0, HALF_H), (HALF_W, HALF_H), (HALF_W, H), (0, H)], dtype=np.int32)
    ZONAS[NOMBRES_ZONAS[3]] = np.array([(HALF_W, HALF_H), (W, HALF_H), (W, H), (HALF_W, H)], dtype=np.int32)
    
    print("Dimensiones de la ventana: {}x{}. Zonas 2x2 definidas.".format(W, H))
    
    return model, cap, writer, ZONAS, TIEMPO_DE_MUESTREO_CORREGIDO, TOTAL_FRAMES


##################################
# --- PROCESAMIENTO DE FRAMES ---
##################################

def procesar_frame(im0, model, TRACKER_CONFIG, CLASES_DE_INTERES, UMBRAL_CONFIANZA, 
                          RESOLUCION_FOTOGRAMA, ZONAS, TIEMPO_DE_MUESTREO_CORREGIDO, tiempo_permanencia,
                          GUARDAR_HIL_ID, idLog_contador, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG, NOMBRES_ZONAS):
    """Realiza la detección, tracking, cálculo de permanencia y etiqueta el frame."""

    # Configuramos los parámetros del seguimiento de objetos y el tracker
    results = model.track(
        im0, persist=True, tracker=TRACKER_CONFIG, classes=CLASES_DE_INTERES,
        verbose=False, conf=UMBRAL_CONFIANZA, imgsz=RESOLUCION_FOTOGRAMA, save=False
    )
    
    im0_etiquetada = results[0].plot()  # contiene el bbox, clase objeto detectado, % confianza deteccion de objeto, track_id

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().tolist()    # recupera los id detectados
        bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # recupera los bbox detectados
        
        for track_id, bbox in zip(track_ids, bboxes):
            
            # Calcular el centro inferior de la bbox
            x_centro = (bbox[0] + bbox[2]) // 2
            y_inferior= bbox[3]
            punto_pie_virtual = (x_centro, y_inferior)
            
            # Verificar en qué zona se encuentra
            zona_actual_nombre = None
            for zona_nombre, poligono in ZONAS.items():
                if is_punto_en_zona(punto_pie_virtual, poligono):
                    zona_actual_nombre = zona_nombre
                    break 
            
            # Actualizar contadores si se encontró una zona
            if zona_actual_nombre:
                # ACUMULACIÓN BASADA EN EL TIEMPO TEÓRICO EQUIVALENTE DEL FRAME POR SI HA HABIDO FRAMES IGNORADOS
                tiempo_permanencia[track_id][zona_actual_nombre] += TIEMPO_DE_MUESTREO_CORREGIDO
                cv2.putText(im0_etiquetada, zona_actual_nombre.split(' - ')[0], (x_centro - 40, y_inferior + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            
            # --- CAPTURAS BBOX en HIL_ID ---
            # Comprueba si se desea que se guarden las capturas de bbox para revision manual (HIL)
            if GUARDAR_HIL_ID:
                idLog_contador = guardar_recorte_y_log(
                    im0, bbox, track_id, idLog_contador, tiempo_permanencia, 
                    NOMBRES_ZONAS, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG
                )
            

    # Dibujar los polígonos de zona en el fotograma etiquetado
    for zona_nombre, poligono in ZONAS.items():
        cv2.polylines(im0_etiquetada, [poligono], isClosed=True, color=(0, 255, 0), thickness=2)
        display_name = zona_nombre.split(' - ')[0]
        cv2.putText(im0_etiquetada, display_name, tuple(poligono[0] + [5, 20]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return im0_etiquetada, tiempo_permanencia, idLog_contador



######################################
# --- MAIN PRINCIPAL DEL PROTOTIPO ---
######################################

def main():

    cap = None
    writer = None
    tiempo_permanencia = defaultdict(lambda: {nombre: 0 for nombre in NOMBRES_ZONAS}) 
    totalFramesIgnorados = 0 
    tiempo_previo = time.time() # usado para el cálculo de FPS de rendimiento
    frame_contador = 0
    idLog_contador = 1 
    TOTAL_FRAMES = 0

    hora_inicio_procesamiento = time.gmtime
    hora_fin_procesamiento = None

    # --- INICIALIZACIÓN ---
    try:
        # TIEMPO_DE_MUESTREO_CORREGIDO 
        # contiene el tiempo real en segundos representado por cada frame procesado
        model, cap, writer, ZONAS, TIEMPO_DE_MUESTREO_CORREGIDO, TOTAL_FRAMES = inicializar_sistema(
            SOURCE, MODELO_DETECCION, OUTPUT_VIDEO_FILE, NOMBRES_ZONAS, GENERAR_VIDEO_ETIQUETADO, 
            GUARDAR_HIL_ID, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG, FRAMES_IGNORADOS
        )
    except ValueError as e:
        print("Se produzco un error: {}".format(e))
        return

    # --- BUCLE PRINCIPAL DE PROCESAMIENTO DE VIDEO ---
    try:

        hora_inicio_procesamiento = time.time()  # para calcular el tiempo de procesamiento del algoritmo

        while cap.isOpened():
            success, im0 = cap.read() 
            if not success:
                print("Fin del video o error en la lectura.")
                break 
            
            frame_contador += 1
            
            # ----------------------------------------------------
            # SALTEO DE FRAMES si así ha sido establecido
            if FRAMES_IGNORADOS > 0 and (frame_contador % (FRAMES_IGNORADOS + 1) != 0):
                if GENERAR_VIDEO_ETIQUETADO:
                    writer.write(im0)
                totalFramesIgnorados += 1
                continue 
            # ----------------------------------------------------
            
            # CÁLCULO DE FPS DE RENDIMIENTO DE LA CPU
            tiempo_actual = time.time()
            intervalo = tiempo_actual - tiempo_previo 
            fps_text = "FPS: {:.2f}".format(1/intervalo) if intervalo > 0 else "FPS: Calculando..."
            tiempo_previo = tiempo_actual


            # --- PROGRESO DE PROCESAMIENTO DELVIDEO ---
            progreso_text = ""
            if TOTAL_FRAMES > 0:    # Si conocemos los frames a procesar significa que no es webcam y estamos capturando un video
                porcentaje = (frame_contador / TOTAL_FRAMES) * 100
                progreso_text = f"Progreso: {porcentaje:.1f}% ({frame_contador}/{TOTAL_FRAMES})"
            
            # --- PROCESAMIENTO DEL FRAME ---
            im0_etiquetada, tiempo_permanencia, idLog_contador = procesar_frame(
                im0, model, TRACKER_CONFIG, CLASES_DE_INTERES, UMBRAL_CONFIANZA, 
                RESOLUCION_FOTOGRAMA, ZONAS, TIEMPO_DE_MUESTREO_CORREGIDO, tiempo_permanencia, 
                GUARDAR_HIL_ID, idLog_contador, OUTPUT_HIL_DIR, OUTPUT_HIL_LOG, NOMBRES_ZONAS
            )
            
            altura_frame = im0_etiquetada.shape[0]    
            
            # Dibujar el texto FPS en la imagen
            cv2.rectangle(im0_etiquetada, (5, altura_frame - 75), (180, altura_frame - 45), (0, 0, 0), -1)    # fondo FPS
            cv2.putText(im0_etiquetada, fps_text, (10, altura_frame - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar progreso procesamiento en pantalla, si se captura desde un video
            if progreso_text:
                    cv2.rectangle(im0_etiquetada, (5, altura_frame - 40), (500, altura_frame), (0, 0, 0), -1)   # fondo progeso
                    cv2.putText(im0_etiquetada, progreso_text, (10, altura_frame - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Escribir fotograma etiquetada, si la generación del video etiquetado está activada
            if GENERAR_VIDEO_ETIQUETADO:
                writer.write(im0_etiquetada)

            # Estadísticas en tiempo real por consola, si la opción está activada 
            if PRINT_CONSOLA:
                dibujar_estadisticas_consola(tiempo_permanencia, NOMBRES_ZONAS, fps_text, progreso_text)
            
            # Visualización del procesamiento de etiquetadao en tiempo real , si la opcion está activada
            if PRINT_PANTALLA:
                cv2.namedWindow(TITULO_VENTANA, cv2.WINDOW_NORMAL)
                cv2.imshow(TITULO_VENTANA, im0_etiquetada)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Si se presiona la tecla "q" el proceso se detiene
                    print("\n--- El usuario interrumpió el procesamiento. Guardando progreso... ---")
                    break 

    except KeyboardInterrupt:
        print("\n--- El usuario interrumpió el procesamiento desde consola (Ctrl+C). Guardando progreso... ---")
        
    #######################################################
    # --- GUARDAR ESTADÍSTICAS Y LIBERACIÓN DE RECURSOS ---
    #######################################################
    finally:

        hora_fin_procesamiento = time.time()   # para calcular el tiempo del procesamiento del algoritmo

        # --- CÁLCULO DE RESUMEN FINAL ---
        if hora_inicio_procesamiento is not None:
            tiempo_total_segundos = hora_fin_procesamiento - hora_inicio_procesamiento            
            
            if tiempo_total_segundos > 0:
                fps_medio_proceso = frame_contador / tiempo_total_segundos
            else:
                fps_medio_proceso = 0
            
            print("\n" + "="*40)
            print("       RESUMEN DE PROCESAMIENTO       ")
            print("="*40)
            print(f"FPS original de captura: {FPS_VIDEO:.2f} FPS")
            print(f"Resolución fotogramas: {RESOLUCION_FOTOGRAMA} píxeles")
            print(f"Tiempo total de ejecución: {tiempo_total_segundos:.2f} segundos")
            print(f"Frames totales procesados: {frame_contador}")
            print(f"Frames ignorados (saltados): {totalFramesIgnorados}")
            print(f"Velocidad media de proceso: {fps_medio_proceso:.2f} FPS")
            print(f"Umbral de confianza de clase de objeto: {UMBRAL_CONFIANZA:.2f} ")
            print("="*40 + "\n")



        if cap is not None:
            cap.release()
            
        if GENERAR_VIDEO_ETIQUETADO and writer is not None:
            writer.release() 
        
        if PRINT_PANTALLA:
            cv2.destroyAllWindows()
        
        guardar_csv(tiempo_permanencia, NOMBRES_ZONAS, OUTPUT_CSV_FILE, totalFramesIgnorados)
        print("FIN DEL PROCESAMIENTO, los recursos han sido liberados.")

if __name__ == "__main__":
    main()
