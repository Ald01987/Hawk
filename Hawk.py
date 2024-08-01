import pandas as pd
import logging
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Configurare il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Caricare il file Excel e i fogli di lavoro
file_path = 'Dashboard_FTTH.xlsx'

try:
    df_ftth = pd.read_excel(file_path, sheet_name='Dashboard_FTTH', header=2)
    df_multi = pd.read_excel(file_path, sheet_name='Indirizzi_Multifibra', header=0)
    df_collab = pd.read_excel(file_path, sheet_name='Collaborazioni', header=0)
    logging.info("Dati caricati con successo")
except Exception as e:
    logging.error(f"Errore nel caricamento del file Excel: {e}")
    df_ftth = pd.DataFrame()
    df_multi = pd.DataFrame()
    df_collab = pd.DataFrame()

# Preprocessare i dati
df_ftth.fillna('', inplace=True)
df_multi.fillna('', inplace=True)
df_collab.fillna('', inplace=True)

def identify_building_type_and_info(row):
    building_type_mapping = {
        'palazzo': 'Palazzo',
        'costruzione indipendente': 'Costruzione indipendente',
        'condominio': 'Condominio',
        'edificio con numero appartam >:8.piano>3': 'Edificio con numero appartam >:8.Piano>3',
        'edificio con numero appartam < 3': 'Edificio con numero appartam < 3',
        'villa': 'Villa',
        'att.comm': 'Att.Comm',
        'quarto piano': 'Quarto piano',
        'primo piano': 'Primo piano',
        'secondo piano': 'Secondo piano',
        'terzo piano': 'Terzo piano',
        'quinto piano': 'Quinto piano',
        'piano 1': 'Piano 1',
        'piano 2': 'Piano 2',
        'piano 3': 'Piano 3',
        'piano 4': 'Piano 4',
        'piano 5': 'Piano 5',
        'piano 6': 'Piano 6',
        '1o piano': '1o piano',
        '2o piano': '2o piano',
        '3o piano': '3o piano',
        '4o piano': '4o piano',
        '5o piano': '5o piano',
        '6o piano': '6o piano',
        'Casa singola': 'Casa singola',
        '1 piano': '1 piano',
        '2 piano': '2 piano',
        '3 piano': '3 piano',
        '4 piano': '4 piano',
        '5 piano': '5 piano',
        'piano 6': '6 piano'
    }
    
    narrative = str(row.get('ACT_NARRATIVE', '')).lower().strip()
    building_type = next((v for k, v in building_type_mapping.items() if k in narrative), 'Altro')

    additional_info_mapping = {
        'palificazione': 'Palificazione',
        'attraversamento stradale': 'Attraversamento Stradale',
        'facciata': 'Facciata',
        'due pezzi di scala': 'Due Pezzi di Scala',
        'più pezzi di scala': 'Più Pezzi di Scala',
        'canalina ostruita': 'Canalina Ostruita',
        'pozzetto': 'Pozzetto',
        'pozzetti': 'Pozzetti',
        'tubazione ostruita': 'Tubazione Ostruita',
        'intercapedine': 'Intercapedine'
    }
    
    notes = str(row.get('NOTE_TECNICO', '')).lower()
    additional_info = [v for k, v in additional_info_mapping.items() if k in notes]

    return building_type, additional_info

def calculate_success_rate(address, city):
    logging.info(f"Calcolo del tasso di successo per indirizzo: {address}, città: {city}")
    
    logging.debug(f"Colonne disponibili in df_ftth: {df_ftth.columns.tolist()}")
    logging.debug(f"Prime righe di df_ftth: {df_ftth.head()}")
    
    filtered_data = df_ftth[(df_ftth['STREET'] == address) & (df_ftth['CITY'] == city)]
    logging.debug(f"Dati filtrati:\n{filtered_data}")
    
    total = len(filtered_data)
    
    if total == 0:
        logging.warning("Nessun dato trovato per l'indirizzo e la città forniti.")
        return {
            'success_rate': 0,
            'details': {},
            'collaboration': 'Indeterminato',
            'multifibra': 'No',
            'building_type': 'Non specificato',
            'additional_info': [],
            'notes_technical': [],
            'management_dates': [],
            'total_rows': 0,
            'success_rows': 0
        }

    success_count = len(filtered_data[filtered_data['CAUSALE'] == 'COMPLWR'])
    success_rate = (success_count / total) * 100
    
    causali_counts = filtered_data['CAUSALE'].value_counts(normalize=True) * 100
    causali_details = causali_counts.to_dict()

    collaboration_value = filtered_data['WR_COLLABORATION'].mode().get(0, 'Indeterminato')
    collaboration = 'Collaborazione' if collaboration_value == 'SI' else 'Singolista'

    has_multifibra = address in df_multi['STREET_MULTI'].values
    if has_multifibra:
        success_rate *= 1.25

    building_type, additional_info = identify_building_type_and_info(filtered_data.iloc[0])

    notes_technical = filtered_data['NOTE_TECNICO'].tolist()
    management_dates = filtered_data['DAT_GIORNO'].tolist()

    return {
        'success_rate': success_rate,
        'details': causali_details,
        'collaboration': collaboration,
        'multifibra': 'Sì' if has_multifibra else 'No',
        'building_type': building_type,
        'additional_info': additional_info,
        'notes_technical': notes_technical,
        'management_dates': management_dates,
        'total_rows': total,
        'success_rows': success_count
    }

def create_pie_chart(details):
    fig = go.Figure()
    if details:
        fig.add_trace(go.Pie(
            labels=list(details.keys()),
            values=list(details.values()),
            hole=0.3,
            textinfo='label+percent',
            marker=dict(colors=['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'])
        ))
        fig.update_layout(
            title_text='Distribuzione delle Causali',
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=True,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        fig.update_traces(
            textfont_size=14,
            pull=[0.1 if val == max(details.values()) else 0 for val in details.values()],
            marker=dict(
                line=dict(
                    color='white',
                    width=2
                )
            )
        )
    return fig

app = Dash(__name__)

app.layout = html.Div([
    html.Link(href='https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap', rel='stylesheet'),
    
    html.H1("Hawk", style={'textAlign': 'center', 'font-family': 'Open Sans'}),
    
    html.Div([
        html.Div([
            html.Label("Città:", style={'fontSize': '18px', 'fontWeight': 'bold', 'font-family': 'Open Sans'}),
            dcc.Dropdown(
                id='city-dropdown',
                options=[{'label': city, 'value': city} for city in df_ftth['CITY'].unique()],
                value=df_ftth['CITY'].unique()[0],
                style={'width': '100%', 'font-family': 'Open Sans'}
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Indirizzo:", style={'fontSize': '18px', 'fontWeight': 'bold', 'font-family': 'Open Sans'}),
            dcc.Dropdown(
                id='address-dropdown',
                options=[{'label': address, 'value': address} for address in df_ftth['STREET'].unique()],
                value=df_ftth['STREET'].unique()[0],
                style={'width': '100%', 'font-family': 'Open Sans'}
            )
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'padding': '10px'}),
    
    html.Div(id='info-container', style={'padding': '10px'}),
    
    dcc.Graph(id='pie-chart')
])

@app.callback(
    [Output('info-container', 'children'),
     Output('pie-chart', 'figure')],
    [Input('city-dropdown', 'value'),
     Input('address-dropdown', 'value')]
)
def update_output(city, address):
    result = calculate_success_rate(address, city)
    
    # Generare il contenuto informativo
    info = html.Div([
        html.H2(f"Dettagli per {address} in {city}", style={'font-family': 'Open Sans'}),
        html.P(f"Tasso di successo: {result['success_rate']:.2f}%", style={'font-family': 'Open Sans'}),
        html.P(f"Collaborazione: {result['collaboration']}", style={'font-family': 'Open Sans'}),
        html.P(f"Multifibra: {result['multifibra']}", style={'font-family': 'Open Sans'}),
        html.P(f"Tipo di edificio: {result['building_type']}", style={'font-family': 'Open Sans'}),
        
        html.Div([
            html.H3("Informazioni Aggiuntive", style={'font-family': 'Open Sans'}),
            html.Ul([html.Li(info) for info in result['additional_info']], style={'font-family': 'Open Sans'})
        ], style={'padding': '10px'}),
        
        html.Div([
            html.H3("NOTE TECNICO", style={'font-family': 'Open Sans'}),
            html.Ul([html.Li(note) for note in result['notes_technical']], style={'font-family': 'Open Sans'})
        ], style={'padding': '10px'}),
        
        html.Div([
            html.H3("Data di gestione", style={'font-family': 'Open Sans'}),
            html.Ul([html.Li(date.strftime('%Y-%m-%d')) for date in result['management_dates']], style={'font-family': 'Open Sans'})
        ], style={'padding': '10px'})
    ])
    
    # Generare il grafico a torta
    pie_chart_figure = create_pie_chart(result['details'])
    
    return info, pie_chart_figure

if __name__ == '__main__':
    app.run_server(debug=True)



# Versione definitiva!
