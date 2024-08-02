# Importa le librerie necessarie
import pandas as pd  # Libreria per la manipolazione dei dati
import logging  # Libreria per la gestione dei log
from dash import Dash, dcc, html  # Moduli per creare l'applicazione Dash
from dash.dependencies import Input, Output, State  # Moduli per gestire i callback di Dash
import plotly.graph_objects as go  # Libreria per la creazione di grafici interattivi
from difflib import get_close_matches  # Modulo per trovare indirizzi simili

# Configurare il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Configura il sistema di logging per registrare informazioni e errori

# Caricare il file Excel e i fogli di lavoro
file_path = 'C:/Users/07703536/Desktop/Dashboard_FTTH/Dashboard_FTTH.xlsx'

try:
    # Carica i dati dai fogli di lavoro Excel
    df_ftth = pd.read_excel(file_path, sheet_name='Dashboard_FTTH', header=2)
    df_multi = pd.read_excel(file_path, sheet_name='Indirizzi_Multifibra', header=0)
    df_pte = pd.read_excel(file_path, sheet_name='Indirizzi_PTE', header=0)
    logging.info("Dati caricati con successo")
except Exception as e:
    # Gestisce eventuali errori nel caricamento del file Excel
    logging.error(f"Errore nel caricamento del file Excel: {e}")
    df_ftth = pd.DataFrame()  # Crea un DataFrame vuoto in caso di errore
    df_multi = pd.DataFrame()  # Crea un DataFrame vuoto in caso di errore
    df_pte = pd.DataFrame()  # Crea un DataFrame vuoto in caso di errore

# Preprocessare i dati
df_ftth.fillna('', inplace=True)  # Sostituisce i valori NaN con stringhe vuote
df_multi.fillna('', inplace=True)  # Sostituisce i valori NaN con stringhe vuote
df_pte.fillna('', inplace=True)  # Sostituisce i valori NaN con stringhe vuote

# Funzione per identificare il tipo di edificio e informazioni aggiuntive
def identify_building_type_and_info(row):
    # Mappa per identificare il tipo di edificio
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
    
    # Estrae e pulisce la narrativa dell'attività
    narrative = str(row.get('ACT_NARRATIVE', '')).lower().strip()
    # Trova il tipo di edificio corrispondente nella mappa
    building_type = next((v for k, v in building_type_mapping.items() if k in narrative), 'Altro')

    # Mappa per identificare informazioni aggiuntive
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
    
    # Estrae e pulisce le note tecniche
    notes = str(row.get('NOTE_TECNICO', '')).lower()
    # Trova le informazioni aggiuntive corrispondenti nella mappa
    additional_info = [v for k, v in additional_info_mapping.items() if k in notes]

    return building_type, additional_info

# Funzione per trovare l'indirizzo PTE più vicino
def get_closest_pte_address(address, pte_df):
    addresses = pte_df['STREET_PTE'].tolist()  # Lista degli indirizzi PTE
    # Trova l'indirizzo più simile usando get_close_matches
    closest_match = get_close_matches(address, addresses, n=1, cutoff=0.7)
    return closest_match[0] if closest_match else 'Nessun indirizzo trovato'

# Funzione per aggiornare il tasso di successo basato sull'indirizzo PTE
def update_success_rate_based_on_pte(address, pte_df, current_rate):
    pte_address = get_closest_pte_address(address, pte_df)  # Trova l'indirizzo PTE più vicino
    logging.info(f"Indirizzo PTE più vicino: {pte_address}")

    pte_row = pte_df[pte_df['STREET_PTE'] == pte_address]  # Filtra i dati per l'indirizzo PTE trovato
    
    if not pte_row.empty:
        pte_address = pte_row.iloc[0]['STREET_PTE']
        aparato_ubicazione = pte_row.iloc[0]['Apparato_UBICAZIONE']

        # Modifica il tasso di successo basato sull'ubicazione dell'apparato
        if aparato_ubicazione in ['ANDRONE', 'INCASSATO', 'INTERNO GARAGE', 'INTERNO INGRESSO', 'PIANO 1', 'SEMINTERRATO', 'SOTTOSCALA']:
            current_rate *= 1.10  # Aumenta del 10%
        else:
            current_rate *= 0.90  # Diminuisci del 10%
        
        logging.info(f"Aggiornamento del tasso di successo: {current_rate:.2f}")
    else:
        logging.warning(f"Nessun indirizzo PTE trovato per l'indirizzo: {address}")

    return pte_address, aparato_ubicazione, current_rate

# Funzione per calcolare il tasso di successo
def calculate_success_rate(address, city):
    logging.info(f"Calcolo del tasso di successo per indirizzo: {address}, città: {city}")
    
    if address in df_multi['STREET_MULTI'].values:
        # Filtra i dati per l'indirizzo e la città
        filtered_data = df_ftth[(df_ftth['STREET'] == address) & (df_ftth['CITY'] == city)]
        
        if filtered_data.empty:
            logging.warning("Indirizzo non presente nello storico delle lavorazioni.")
            
            # Se l'indirizzo non è presente, calcola la percentuale di successo basata sull'indirizzo PTE
            pte_address, aparato_ubicazione, base_success_rate = update_success_rate_based_on_pte(address, df_pte, 30)

            has_multifibra = address in df_multi['STREET_MULTI'].values
            if has_multifibra:
                base_success_rate *= 1.25

            return {
                'success_rate': base_success_rate,
                'details': {},
                'collaboration': 'Indeterminato',
                'multifibra': 'Sì' if has_multifibra else 'No',
                'building_type': 'Non specificato',
                'additional_info': ['Indirizzo non presente nello storico delle lavorazioni'],
                'notes_technical': [],
                'management_dates': [],
                'total_rows': 0,
                'success_rows': 0,
                'pte_address': pte_address,
                'apparato_ubicazione': aparato_ubicazione
            }

        total = len(filtered_data)  # Numero totale di righe
        success_count = len(filtered_data[filtered_data['CAUSALE'] == 'COMPLWR'])  # Conteggio righe di successo
        success_rate = (success_count / total) * 100  # Calcolo della percentuale di successo
        
        # Dettagli delle causali
        causali_counts = filtered_data['CAUSALE'].value_counts(normalize=True) * 100
        causali_details = causali_counts.to_dict()

        # Determina il tipo di collaborazione
        collaboration_value = filtered_data['WR_COLLABORATION'].mode().get(0, 'Indeterminato')
        collaboration = 'Collaborazione' if collaboration_value == 'SI' else 'Singolista'

        # Verifica se l'indirizzo è multifibra
        has_multifibra = address in df_multi['STREET_MULTI'].values
        if has_multifibra:
            success_rate *= 1.25

        # Identifica il tipo di edificio e le informazioni aggiuntive
        building_type, additional_info = identify_building_type_and_info(filtered_data.iloc[0])

        # Estrai le note tecniche e le date di gestione
        notes_technical = filtered_data['NOTE_TECNICO'].tolist()
        management_dates = pd.to_datetime(filtered_data['DAT_GIORNO']).tolist()

        # Aggiorna il tasso di successo basato sull'indirizzo PTE
        pte_address, aparato_ubicazione, success_rate = update_success_rate_based_on_pte(address, df_pte, success_rate)

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
            'success_rows': success_count,
            'pte_address': pte_address,
            'apparato_ubicazione': aparato_ubicazione
        }
    else:
        logging.warning("Indirizzo non trovato nella colonna STREET_MULTI.")
        
        # Elaborazione alternativa per indirizzi non presenti in STREET_MULTI
        pte_address, aparato_ubicazione, base_success_rate = update_success_rate_based_on_pte(address, df_pte, 0)
        
        has_multifibra = address in df_multi['STREET_MULTI'].values
        if has_multifibra:
            base_success_rate *= 1.25

        return {
            'success_rate': base_success_rate,
            'details': {},
            'collaboration': 'Indeterminato',
            'multifibra': 'Sì' if has_multifibra else 'No',
            'building_type': 'Non specificato',
            'additional_info': ['Indirizzo non presente nello storico delle lavorazioni'],
            'notes_technical': [],
            'management_dates': [],
            'total_rows': 0,
            'success_rows': 0,
            'pte_address': pte_address,
            'apparato_ubicazione': aparato_ubicazione
        }

# Funzione per creare un grafico a torta delle causali
def create_pie_chart(causali_details):
    fig = go.Figure()
    if causali_details:
        # Aggiunge un grafico a torta con le causali
        fig.add_trace(go.Pie(labels=list(causali_details.keys()), values=list(causali_details.values()), hole=0.4))
        fig.update_layout(
            title_text='Distribuzione Causali',
            annotations=[dict(text='Causali', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
    else:
        # Grafico a torta con una sola voce "Nessuna" se non ci sono dati
        fig.add_trace(go.Pie(labels=['Nessuna'], values=[1], hole=0.4))
        fig.update_layout(
            title_text='Distribuzione Causali',
            annotations=[dict(text='Nessuna', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )

    return fig

# Creare l'app Dash
app = Dash(__name__)

# Definire il layout dell'app
app.layout = html.Div([
    html.H1("HAWK", style={'text-align': 'center','color': '#008080','font-family': 'Arial, sans-serif'}),
    
    html.Div(
        "Per la verifica dell'indirizzo ti ricordo di utilizzare solo caratteri maiuscoli edi utilizzare il browser Google Chrome. Grazie.",
        style={'color': '#dc143c', 'text-align': 'left', 'font-size': '18px', 'margin-bottom': '10px','font-family': 'Arial, sans-serif'}
    ),
    
    # Dropdown per selezionare la città
    dcc.Dropdown(
        id='city-dropdown',
        options=[{'label': city, 'value': city} for city in df_ftth['CITY'].unique()],
        value=df_ftth['CITY'].unique()[0],  # Impostare un valore predefinito
        style={'width': '50%', 'margin-bottom': '3px','font-family': 'Arial, sans-serif'}
    ),
    
    # Input per inserire l'indirizzo
    dcc.Input(
        id='address-input',
        type='text',
        value='',
        placeholder='Inserisci l\'indirizzo',
        style={'width': '50%', 'padding': '10px', 'margin-bottom': '5px','font-family': 'Arial, sans-serif'}
    ),
    
    # Bottone per avviare la ricerca
    html.Button(
        'Cerca',
        id='search-button',
        n_clicks=0,
        style={
            'width': '6%',
            'padding': '10px',
            'background-color': '#007BFF',
            'color': 'white',
            'border': 'none',
            'border-radius': '5px',
            'cursor': 'pointer',
            'font-size': '16px',
            'font-family': 'Arial, sans-serif'
        }
    ),
    
    # Contenitore per visualizzare le informazioni
    html.Div(id='info-container', style={'margin-top': '20px'}),
    
    # Grafico a torta
    dcc.Graph(id='pie-chart', style={'margin-top': '20px'})
])

# Definire il callback per aggiornare l'output in base ai click del bottone di ricerca
@app.callback(
    [Output('info-container', 'children'),
     Output('pie-chart', 'figure')],
    [Input('search-button', 'n_clicks')],
    [State('address-input', 'value'),
     State('city-dropdown', 'value')]
)
def update_output(n_clicks, address, city):
    if n_clicks > 0:
        try:
            # Calcola il tasso di successo e ottieni i risultati
            result = calculate_success_rate(address, city)
            
            # Crea la visualizzazione delle informazioni
            info = html.Div([
                html.H2(f"Dettagli per {address}", style={'font-family': 'Arial, sans-serif','color':'#1e90ff'}),
                html.P(f"Tasso di successo: {result['success_rate']:.2f}%", style={'font-family': 'Arial, sans-serif'}),
                html.P(f"Collaborazione: {result['collaboration']}", style={'font-family': 'Arial, sans-serif'}),
                html.P(f"Multifibra: {result['multifibra']}", style={'font-family': 'Arial, sans-serif'}),
                html.P(f"Tipo di edificio: {result['building_type']}", style={'font-family': 'Arial, sans-serif'}),
                html.P(f"Indirizzo PTE: {result['pte_address']}", style={'font-family': 'Arial, sans-serif'}),
                html.P(f"Apparato Ubicazione: {result['apparato_ubicazione']}", style={'font-family': 'Arial, sans-serif'}),
                
                html.Div([
                    html.H3("Informazioni Aggiuntive", style={'font-family': 'Arial, sans-serif','color':'#1e90ff'}),
                    html.Ul([html.Li(info) for info in result['additional_info']], style={'font-family': 'Arial, sans-serif'})
                ], style={'padding': '10px'}),
                
                html.Div([
                    html.H3("NOTE TECNICO", style={'font-family': 'Arial, sans-serif','color':'#1e90ff'}),
                    html.Ul([html.Li(note) for note in result['notes_technical']], style={'font-family': 'Arial, sans-serif'})
                ], style={'padding': '10px'}),
                
                html.Div([
                    html.H3("Data di gestione", style={'font-family': 'Arial, sans-serif','color':'#1e90ff'}),
                    html.Ul([html.Li(date.strftime('%Y-%m-%d')) for date in result['management_dates']], style={'font-family': 'Arial, sans-serif'})
                ], style={'padding': '10px'})
            ])
            
            # Crea il grafico a torta
            pie_chart_figure = create_pie_chart(result['details'])
            
            return info, pie_chart_figure
        except Exception as e:
            logging.error(f"Errore nell'aggiornamento della dashboard: {e}")
            return html.Div(["Errore nel calcolo dei dati. Verifica i log per dettagli."]), go.Figure()
    return html.Div(), go.Figure()

# Avvia l'app Dash
if __name__ == '__main__':
    app.run_server(debug=True)

