import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)

# Caricare il file Excel e i fogli di lavoro
file_path = 'C:\Users\07703536\Desktop\Dashboard_FTTH\Dashboard_FTTH.xlsx'

try:
    df_ftth = pd.read_excel(file_path, sheet_name='Dashboard_FTTH', header=2)
    df_multi = pd.read_excel(file_path, sheet_name='Indirizzi_Multifibra')
    logging.info("Dati caricati con successo")
except Exception as e:
    logging.error(f"Errore nel caricamento del file Excel: {e}")
    df_ftth = pd.DataFrame()
    df_multi = pd.DataFrame()

# Preprocessare i dati
df_ftth.fillna('', inplace=True)
df_multi.fillna('', inplace=True)

def identify_building_type_and_info(row):
    building_type = 'Altro'
    additional_info = []

    narrative = str(row['ACT_NARRATIVE']).lower().strip()

    # Identificare il tipo di edificio
    if 'palazzo' in narrative:
        building_type = 'Palazzo'
    elif 'costruzione indipendente' in narrative:
        building_type = 'Costruzione indipendente'
    elif 'condominio' in narrative:
        building_type = 'Condominio'
    elif 'edificio con numero appartam >:8.piano>3' in narrative:
        building_type = 'Edificio con numero appartam >:8.Piano>3'
    elif 'edificio con numero appartam < 3' in narrative:
        building_type = 'Edificio con numero appartam < 3'
    elif 'villa' in narrative:
        building_type = 'Villa'
    elif 'att.comm' in narrative:
        building_type = 'Att.Comm'
    elif 'quarto piano' in narrative:
        building_type = 'Quarto piano'
    elif 'primo piano' in narrative:
        building_type = 'Primo piano'
    elif 'secondo piano' in narrative:
        building_type = 'Secondo piano'
    elif 'terzo piano' in narrative:
        building_type = 'Terzo piano'
    elif 'quinto piano' in narrative:
        building_type = 'Quinto piano'
    elif 'piano 1' in narrative:
        building_type = 'Piano 1'
    elif 'piano 2' in narrative:
        building_type = 'Piano 2'
    elif 'piano 3' in narrative:
        building_type = 'Piano 3'
    elif 'piano 4' in narrative:
        building_type = 'Piano 4'
    elif 'piano 5' in narrative:
        building_type = 'Piano 5'
    elif 'piano 6' in narrative:
        building_type = 'Piano 6'
    elif '1o piano' in narrative:
        building_type = '1o piano'
    elif '2o piano' in narrative:
        building_type = '2o piano'
    elif '3o piano' in narrative:
        building_type = '3o piano'
    elif '4o piano' in narrative:
        building_type = '4o piano'
    elif '5o piano' in narrative:
        building_type = '5o piano'
    elif '6o piano' in narrative:
        building_type = '6o piano'   
    elif 'Casa singola' in narrative:
        building_type = 'Casa singola'
    elif '1 piano' in narrative:
        building_type = '1 piano'
    elif '2 piano' in narrative:
        building_type = '2 piano'
    elif '3 piano' in narrative:
        building_type = '3 piano'
    elif '4 piano' in narrative:
        building_type = '4 piano'
    elif '5 piano' in narrative:
        building_type = '5 piano'
    elif 'piano 6' in narrative:
        building_type = '6 piano'
                 
    # Rilevare informazioni aggiuntive
    if 'palificazione' in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Palificazione')
    if 'attraversamento stradale'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Attraversamento Stradale')
    if'facciata'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Facciata')
    if'due pezzi di scala'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Due Pezzi di Scala')
    if'più pezzi di scala'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Più Pezzi di Scala')
    if'canalina ostruita'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('canalina ostruita')
    if'Pozzetto'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Pozzetto')
    if'Pozzetti'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Pozzetti')
    if'Tubazione ostruita'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Tubazione ostruita')
    if'Intercapedine'in str(row['NOTE_TECNICO']).lower():
        additional_info.append('Intercapedine')
            
    if not additional_info:
        additional_info.append('Nessuna informazione aggiuntiva rilevata')
    
    return building_type, additional_info


# Funzione per calcolare i tassi di successo
def calculate_success_rate(address, city):
    logging.info(f"Calcolo del tasso di successo per indirizzo: {address}, città: {city}")
    filtered_data = df_ftth[(df_ftth['STREET'] == address) & (df_ftth['CITY'] == city)]
    total = len(filtered_data)
    if total == 0:
        logging.warning("Nessun dato trovato per l'indirizzo e la città forniti.")
        return {'success_rate': 0, 'details': {}, 'collaboration': 'Indeterminato', 'multifibra': 'No', 'building_type': 'Non specificato', 'additional_info': []}
    
    success_count = len(filtered_data[filtered_data['CAUSALE'] == 'COMPLWR'])
    success_rate = (success_count / total) * 100
    
    causali_counts = filtered_data['CAUSALE'].value_counts(normalize=True) * 100
    causali_details = causali_counts.to_dict()

    if not filtered_data['WR_COLLABORATION'].empty:
        collaboration_value = filtered_data['WR_COLLABORATION'].mode()[0]
        collaboration = 'Collaborazione' if collaboration_value == 'SI' else 'Singolista'
    else:
        collaboration = 'Indeterminato'

    has_multifibra = address in df_multi['STREET MULTI'].values
    if has_multifibra:
        success_rate *= 1.25

    building_type, additional_info = identify_building_type_and_info(filtered_data.iloc[0])

    return {
        'success_rate': success_rate,
        'details': causali_details,
        'collaboration': collaboration,
        'multifibra': 'Sì' if has_multifibra else 'No',
        'building_type': building_type,
        'additional_info': additional_info
    }

# Inizializzare l'app Dash
app = Dash(__name__)

# Layout dell'app
app.layout = html.Div([
    # Link al font di Google Fonts
    html.Link(href='https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap', rel='stylesheet'),
    
    html.H1("Hawk", style={'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'Roboto, sans-serif'}),
    
    html.Div([
        html.Div([
            html.Label("Città:", style={'fontSize': '18px', 'fontWeight': 'bold', 'fontFamily': 'Roboto, sans-serif', 'color': '#2980b9'}),
            dcc.Input(id='city-input', type='text', value='', placeholder='Inserisci città', style={'fontFamily': 'Roboto, sans-serif', 'border': '1px solid #2980b9'}),
            html.Br(),
            html.Label("Indirizzo:", style={'fontSize': '18px', 'fontWeight': 'bold', 'fontFamily': 'Roboto, sans-serif', 'color': '#2980b9'}),
            dcc.Input(id='address-input', type='text', value='', placeholder='Inserisci indirizzo', style={'fontFamily': 'Roboto, sans-serif', 'border': '1px solid #2980b9'}),
            html.Br(),
            html.Button('Verifica', id='submit-button', n_clicks=0, style={'marginTop': '10px', 'padding': '10px 20px', 'backgroundColor': '#2980b9', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'fontFamily': 'Roboto, sans-serif'}),
            html.Div(id='error-message', style={'color': 'red', 'marginTop': '10px', 'fontFamily': 'Roboto, sans-serif'})
        ], style={'width': '40%', 'margin': 'auto'}),
        
        html.Div(id='output-container', style={'marginTop': '20px', 'fontFamily': 'Roboto, sans-serif'}),
        dcc.Graph(id='pie-chart', style={'marginTop': '20px'})
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)'})
])

# Callback per aggiornare il contenitore di output e il grafico a torta
@app.callback(
    [Output('output-container', 'children'),
     Output('pie-chart', 'figure'),
     Output('error-message', 'children')],
    [Input('submit-button', 'n_clicks')],
    [State('city-input', 'value'),
     State('address-input', 'value')]
)
def update_output(n_clicks, city, address):
    logging.info(f"Callback chiamato con {n_clicks} clic, città: {city}, indirizzo: {address}")
    
    if n_clicks > 0:
        if not city or not address:
            return "", {}, "Per favore, inserisci sia la città che l'indirizzo."

        try:
            result = calculate_success_rate(address, city)
            success_rate = result['success_rate']
            details = result['details']
            collaboration = result['collaboration']
            multifibra = result['multifibra']
            building_type = result['building_type']
            additional_info = result['additional_info']

            logging.info(f"Risultati calcolati: {result}")

            # Creare un grafico a dispersione 3D come esempio
            scatter_3d_fig = go.Figure(data=[go.Scatter3d(
                x=df_ftth['COLUMN_X'],  # Modifica con i dati effettivi
                y=df_ftth['COLUMN_Y'],  # Modifica con i dati effettivi
                z=df_ftth['COLUMN_Z'],  # Modifica con i dati effettivi
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_ftth['COLUMN_COLOR'],  # Modifica con i dati effettivi
                    colorscale='Viridis'
                )
            )])

            scatter_3d_fig.update_layout(
                title='Grafico a Dispersione 3D',
                scene=dict(
                    xaxis_title='Asse X',
                    yaxis_title='Asse Y',
                    zaxis_title='Asse Z'
                )
            )

            output_text = html.Div([
                html.P(f"Indirizzo: {address}, {city}", style={'fontSize': '16px', 'fontWeight': 'bold', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
                html.P(f"Tasso di successo: {success_rate:.2f}%", style={'fontSize': '16px', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
                html.P(f"Tipo di lavoro: {collaboration}", style={'fontSize': '16px', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
                html.P(f"Presenza di multifibra: {multifibra}", style={'fontSize': '16px', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
                html.P(f"Tipo di edificio: {building_type}", style={'fontSize': '16px', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
                html.P(f"Informazioni aggiuntive: {', '.join(additional_info)}", style={'fontSize': '16px', 'fontFamily': 'Roboto, sans-serif', 'color': '#34495e'}),
            ], style={'backgroundColor': '#bdc3c7', 'padding': '10px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)'})

            return output_text, fig, ""
        except Exception as e:
            logging.error(f"Errore durante l'aggiornamento dell'output: {e}")
            return f"Errore nell'aggiornamento dell'output: {e}", {}, ""
    
    return "", {}, ""

# Eseguire l'app Dash
if __name__ == '__main__':
    app.run_server(debug=True)