######################################
# --- PROYECTO PARA TFG DE LA UNIR ---
# Autor: Francisco Javier Ortiz Gonzalez
# Fecha: Diciembre, 2025
# Licencia: AGPL_v3
######################################

import csv
import os
from collections import defaultdict

#####################################
# --- PARA LA GENERACIÓN DE RUTAS ---
#####################################
RUTA_COMPLETA_SCRIPT = os.path.abspath(__file__)    #.../TFG/codigo/archivo.py
SCRIPT_DIR = os.path.dirname(RUTA_COMPLETA_SCRIPT)  # subimos un nivel  .../TFG/codigo
RUTA_RAIZ_PROYECTO = os.path.dirname(SCRIPT_DIR)    # subimos un nivel  .../TFG

# Carpeta principal done se almacenan los recortes de bounding box
OUTPUT_HIL_DIR = os.path.join(RUTA_RAIZ_PROYECTO, "HIL_ID") 

# Archivo de log de detecciones
OUTPUT_HIL_LOG = os.path.join(OUTPUT_HIL_DIR, "id_detection_log.csv") 

# Archivo de sealida con los tiempos fusionados
OUTPUT_FUSION_CSV = os.path.join(RUTA_RAIZ_PROYECTO, "estadisticas", "tiempos_id_fusionados.csv")

# Nombres de las zonas en el log 
HEADERS_ZONAS = ["zona1", "zona2", "zona3", "zona4"]


##############################
# --- FUNCIONES AUXILIARES ---
##############################

def mapear_logs_a_id_final(hil_dir):
    """
    Recorre la jerarquia de carpeta y crea un mapa {idLog: idPersona_final}.
    Es decir, las capturas que se encuentren dentro de ID_X, se asocian ahora a X 
    """

    print("Mapeando estructura de carpetas de HIL_ID... ")

    id_mapa = {}
    
    # Recorre todas las subcarpetas en OUTPUT_HIL_DIR 
    for raiz, directorios, archivos in os.walk(hil_dir):
        carpeta_nombre = os.path.basename(raiz)
        
        # Solo procesa carpetas que siguen el formato 'ID_X'
        if not carpeta_nombre.startswith('ID_'):
            continue
            
        try:
            # Obtiene el ID numérico (X) de la carpeta 'ID_X'
            id_persona_final = int(carpeta_nombre.split('_')[1])
        except (IndexError, ValueError):
            continue

        for archivo in archivos:
            if archivo.endswith('.jpg'):
                # El nombre del archivo es el idLog
                id_log_str = archivo.split('.')[0]
                try:
                    id_log = int(id_log_str)
                    id_mapa[id_log] = id_persona_final
                except ValueError:
                    # Ignora archivos con nombres que no sigue el formato esperado
                    continue
                    
    print("Mapeo completado. {} archivos de log encontrados en ID_X.".format(len(id_mapa)))
    return id_mapa


def cargar_y_sumar_tiempos(log_ruta, id_mapa): 
    """
    Carga el log de detecciones y suma los tiempos de permanencia segun datos asociados al idLog
    Utiliza la lógica delta, es decir al usar tiempos acumulados, para tener el valor real necesitamos calcular su diferencia
    Hay que tener en cuenta que el tracker puede haberse "reiniciado" y para la misma persona volver haber empezado desde 0
    """
    
    # El diccionario final que contendrá los tiempos sumados por ID_FINAL 
    # (el que se ha asignado tras revisión humana)
    dic_tiempos_fusionados = defaultdict(lambda: dict.fromkeys(HEADERS_ZONAS, 0.0))
    total_filas_csv = 0
    total_logs_procesados = 0 
    
    if not os.path.exists(log_ruta):
        print("Error: Archivo de log no encontrado en {}".format(log_ruta))
        return dic_tiempos_fusionados

    print("Cargando log desde: {}".format(log_ruta))

    # Diccionario que guarda el último valor acumulado que vimos para cada ID_ORIGINAL del tracker, antes de la revisión humana
    dic_tiempos_originales_tracker = defaultdict(lambda: dict.fromkeys(HEADERS_ZONAS, 0.0))
    
    try:
        with open(log_ruta, mode='r', newline='') as infile:
            reader = csv.DictReader(infile) 
            
            for row in reader:
                total_filas_csv += 1 
                try:
                    # Obtenemos los IDs
                    id_log = int(row['idLog'])
                    id_tracker_original = int(row['idPersona']) # El ID que puso el tracker
                    id_final = id_mapa.get(id_log)              # El ID verificado por humano
                    
                    # Si este idLog no está en el mapa, es que se borró intencionadamente por el humano
                    # en ese caso ignoramos
                    if id_final is None:
                        continue
                    
                    # Calculamos los deltas o diferencias de tiempos por cada zona ---
                    for zona in HEADERS_ZONAS:
                        
                        # Cargar el valor de esta fila del CSV
                        tiempo_str = row[zona].strip().replace(',', '.')
                        valor_actual_del_log = float(tiempo_str)
                        
                        # Obtenemos el valor anterior
                        # Es decir, buscamos el último valor que guardamos para este ID_tracker
                        ultimo_valor_guardado = dic_tiempos_originales_tracker[id_tracker_original][zona]
                        
                        # Calculamos la diferencia de tiempo que ha pasado, detectando los casos de reseteo del tracker ---
                        tiempo_delta = 0.0
                        
                        # Si el tracker se ha reiniciado: el valor actual es MENOR que el anterior
                        # es decir, el tracker reseteó su contador para ese ID.
                        if valor_actual_del_log < ultimo_valor_guardado:
                            # El tiempo a sumar es el valor actual (el inicio desde el reseteo)
                            tiempo_delta = valor_actual_del_log
                        else:
                            # no ha habido reseteo del tracker
                            # el tiempo pasado es el valor actual menos el último que guardamos
                            tiempo_delta = valor_actual_del_log - ultimo_valor_guardado
                            
                        # Actualizamos el diccionario de referencia para próximas iteraciones
                        # para ello debemos actualizar y guardar el valor actual en el diccionario original de referencia
                        # ahora 'valor_actual_del_log' se convierte en el 'ultimo_valor_guardado'
                        dic_tiempos_originales_tracker[id_tracker_original][zona] = valor_actual_del_log
                        
                        # Sumamos el resultado al id_final
                        # es decir, suma el tiempo de este frame (el delta) al ID_FINAL verificado por el humano
                        dic_tiempos_fusionados[id_final][zona] += tiempo_delta
                                
                    total_logs_procesados += 1
                            
                except ValueError as e:
                    print(f"ERROR DE FORMATO en fila {total_filas_csv}. Error: {e}")
                    continue
                except KeyError as e:
                    print(f"ERROR DE ENCABEZADO en fila {total_filas_csv}: Columna {e} no encontrada.")
                    continue

    except Exception as e:
        print(f"Error general al leer el archivo CSV: {e}. Se han procesado {total_filas_csv} filas hasta el error.")
        
    print("\n--- RESUMEN ---")
    print("Filas totales en el log: {}".format(total_filas_csv))
    print("Logs procesados: {}".format(total_logs_procesados))
    print("IDs de persona consolidados: {}".format(len(dic_tiempos_fusionados)))
    print("---------------")
    
    return dic_tiempos_fusionados

def guardar_tiempos_fusionados(tiempos_fusionados, ruta_archivo_fusion, zonas):
    """
    Guarda los tiempos totales de permanencia por ID de persona en un nuevo archivo CSV.
    """
    if not tiempos_fusionados:
        print("No hay datos para guardar.")
        return

    headers = ["idPersona"] + zonas
    
    print("\nGuardando datos fusionados en: {}".format(ruta_archivo_fusion))
    
    try:
        with open(ruta_archivo_fusion, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(headers)
            
            for id_persona, tiempos in sorted(tiempos_fusionados.items()):
                row = [id_persona] + [round(tiempos[zona], 2) for zona in zonas]
                writer.writerow(row)

        print("El proceso de fusión ha terminado.")

    except Exception as e:
        print("Error al crear archivo de fusión: {}".format(e))

###############################
# --- MAIN PRINCIPAL FUSIÓN ---
###############################

def main():
    """Ejecuta el proceso de fusión de tiempos de permanencia por ID, basado en la estructura de disco."""
    
    # Mapea los ids de las capturas 
    id_mapa = mapear_logs_a_id_final(OUTPUT_HIL_DIR)
    
    if not id_mapa:
        print("No se encontraron imágenes en las carpetas ID_X para procesar.")
        return
    
    # Carga y suma los tiempos, usando el mapa para determinar el ID final tras revisión humana
    tiempos_consolidados = cargar_y_sumar_tiempos(OUTPUT_HIL_LOG, id_mapa)
    
    # Guarda el nuevo CSV con los resultados fusionados
    guardar_tiempos_fusionados(tiempos_consolidados, OUTPUT_FUSION_CSV, HEADERS_ZONAS)

if __name__ == "__main__":
    main()
