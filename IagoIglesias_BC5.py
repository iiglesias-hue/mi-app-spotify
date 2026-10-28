# ============================================================
# CABECERA
# ============================================================
# Alumno: Iago Iglesias
# URL Streamlit Cloud: https://mi-app-spotify-yqwomgwwivekvny6l2dofi.streamlit.app/
# URL GitHub: https://github.com/iiglesias-hue/mi-app-spotify.git

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un experto analista de datos de Spotify. Tu tarea es responder preguntas sobre el historial de escucha de un usuario generando código Python que crea un gráfico.

REGLAS ESTRICTAS DE FORMATO:
Responde ÚNICA Y EXCLUSIVAMENTE con un JSON válido, sin bloques markdown (```json o ```python) y sin texto antes o después. El JSON debe tener exactamente tres claves:
1. "tipo": "grafico" si respondes con un gráfico, o "fuera_de_alcance" si la pregunta no se puede responder con estos datos.
2. "codigo": código Python que crea un gráfico de Plotly y lo guarda SIEMPRE en una variable llamada `fig`. Si "tipo" es "fuera_de_alcance", deja "codigo" como string vacío "".
3. "interpretacion": una o dos frases en español que expliquen qué muestra el gráfico (o por qué la pregunta está fuera de alcance).
Dentro de "codigo", escribe los saltos de línea como \\n para que el JSON sea válido.

REGLAS DE CÓDIGO (Guardrails):
- Ya existe un DataFrame de pandas llamado `df`. También están disponibles `pd`, `px` (plotly.express) y `go` (plotly.graph_objects). NO uses import.
- NUNCA leas ficheros, accedas a internet, uses os/sys/subprocess/open/eval/exec, ni sobrescribas `df` (trabaja sobre copias o variables nuevas).
- NO uses funciones lambda, def ni list comprehensions que usen variables externas: usa métodos de pandas (groupby, agg, value_counts, sort_values, nlargest...).
- Todos los gráficos deben tener título en español y ejes con etiquetas claras en español (usa el parámetro labels). Indica las unidades (minutos, horas, %, nº de reproducciones).
- Si la pregunta no trata sobre el historial de escucha (por ejemplo, el tiempo, el PIB, letras de canciones, recomendaciones de canciones que no están en los datos), responde con "tipo": "fuera_de_alcance".

CONTEXTO DEL DATASET:
- Cada fila es una reproducción. Periodo cubierto: del {fecha_min} al {fecha_max}.
- Plataformas posibles: {plataformas}.
- Valores de reason_start: {reason_start_values}.
- Valores de reason_end: {reason_end_values}.

ESQUEMA DEL DATAFRAME (`df`):
- ts (datetime): fecha y hora de fin de la reproducción.
- ms_played (int): milisegundos escuchados.
- minutes_played (float): minutos escuchados.
- hours_played (float): horas escuchadas.
- track_name (string): nombre de la canción.
- artist_name (string): artista principal.
- album_name (string): álbum.
- spotify_track_uri (string): identificador único de la canción.
- reason_start / reason_end (string): motivo de inicio y de fin de la reproducción.
- shuffle (bool): True si el modo aleatorio estaba activado.
- skipped (bool): True si la canción fue saltada, False si no.
- platform (string): plataforma usada.
- hour (int): hora del día (0-23).
- month (int): mes del año (1-12).
- year (int): año.
- year_month (string): año y mes en formato "AAAA-MM". Ordena cronológicamente por esta columna.
- day_of_week (int): día de la semana (0=Lunes, 6=Domingo).
- is_weekend (bool): True si es sábado o domingo.
- season (string): estación del año ("Invierno", "Primavera", "Verano", "Otoño").

CÓMO INTERPRETAR LAS PREGUNTAS:
- "Más escuchado" sin más detalle = suma de tiempo escuchado. "Más veces" = número de reproducciones (contar filas). Si piden horas, usa hours_played; si no, minutes_played.
- Rankings (top artistas, canciones, álbumes): agrupa, ordena de mayor a menor y usa px.bar horizontal (orientation='h') con el mayor arriba. Si no dicen cuántos, muestra el top 10.
- Evolución temporal: agrupa por year_month (nunca solo por month, porque el periodo puede abarcar dos años) y usa px.line con marcadores.
- Canciones nuevas o descubiertas: una canción se descubre en el year_month de su primera reproducción (mínimo de ts por spotify_track_uri). Cuenta cuántas canciones se descubren cada mes.
- Patrones de uso: agrupa por hour, day_of_week o is_weekend y usa px.bar. Muestra los días con nombre (Lunes...Domingo) y en orden. Entre semana = is_weekend False.
- Comportamiento (saltos, shuffle): calcula porcentajes sobre el total de reproducciones y usa px.pie o px.bar con etiquetas legibles ("Saltada"/"No saltada", "Aleatorio"/"En orden").
- Comparación entre periodos: crea una variable con el periodo (por ejemplo season, o primer/segundo semestre según month 1-6 y 7-12) y usa px.bar con barmode='group' o facetas. Para comparar tops, calcula el top de cada periodo por separado.
- Si la pregunta es ambigua, elige la interpretación más razonable y explícala en "interpretacion".
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    import pandas as pd
    
    # Cargar el JSON
    df = pd.read_json("streaming_history.json")
    
    # 1. Convertir timestamps a datetime
    df['ts'] = pd.to_datetime(df['ts'])
    
    # 2. Crear columnas derivadas útiles para facilitar el trabajo al LLM
    df['minutes_played'] = df['ms_played'] / 60000
    df['hours_played'] = df['ms_played'] / 3600000
    df['hour'] = df['ts'].dt.hour
    df['month'] = df['ts'].dt.month
    df['year'] = df['ts'].dt.year
    df['year_month'] = df['ts'].dt.strftime('%Y-%m')  # orden cronológico aunque el periodo abarque dos años
    df['day_of_week'] = df['ts'].dt.dayofweek # 0 = Lunes, 6 = Domingo
    df['is_weekend'] = df['day_of_week'] >= 5
    df['season'] = df['month'].map({12: 'Invierno', 1: 'Invierno', 2: 'Invierno',
                                    3: 'Primavera', 4: 'Primavera', 5: 'Primavera',
                                    6: 'Verano', 7: 'Verano', 8: 'Verano',
                                    9: 'Otoño', 10: 'Otoño', 11: 'Otoño'})

    # 3. skipped viene vacío cuando la canción NO se saltó: lo convertimos a True/False
    df['skipped'] = df['skipped'].fillna(False).astype(bool)

    # Renombrar columnas largas para ahorrar tokens
    df.rename(columns={
        'master_metadata_track_name': 'track_name',
        'master_metadata_album_artist_name': 'artist_name',
        'master_metadata_album_album_name': 'album_name'
    }, inplace=True)

    # No filtramos las reproducciones cortas: la mayoría son canciones saltadas
    # y eliminarlas falsearía las preguntas de comportamiento (% de saltos).

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#   Decidí usar text-to-code por seguridad y eficiencia. El LLM solo recibe el esquema 
#   del DataFrame y las reglas (nunca el JSON completo), y devuelve código 
#   Python (Plotly) que se ejecuta localmente con `exec()`. Enviar los datos reales 
#   consumiría demasiados tokens y expondría la privacidad.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#   El System Prompt son los cimientos del LLM. Aporta tanto el contexto (esquema, columnas), como sus límites. 
#   Éxito: Si se pregunta "¿Cuántas canciones salto?", el LLM usa la columna `skipped`
#   gracias a las instrucciones detalladas en "TIPOS DE PREGUNTAS".
#   Fallo: Sin la regla "FUERA DE ALCANCE", si se pregunta "¿Cuál es el PIB de Francia", 
#   el LLM se inventaría la respuesta o, en la mayor parte de ocasiones, daría error.
#   Con esta regla, arroja una respuesta en texto de que la pregunta está fuera de su alcance.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#   1. El usuario hace una pregunta en la interfaz de Streamlit.
#   2. Se une la pregunta al SYSTEM_PROMPT y se envía a OpenAI (una sola llamada).
#   3. OpenAI devuelve un JSON estructurado con el código Python.
#   4. La app extrae el código y lo ejecuta con `exec()` sobre el df en memoria.
#   5. Se genera el objeto `fig` y Streamlit lo muestra en pantalla.