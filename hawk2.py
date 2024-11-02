import os
import re
import logging
import numpy as np
import pandas as pd
import unicodedata
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
import lightgbm as lgb
import numpy
import uuid
import json
from imblearn.over_sampling import SMOTE
from collections import defaultdict
import joblib
from datetime import datetime
from pathlib import Path
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    make_scorer,
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score,
    log_loss, 
    brier_score_loss
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
import logging

# Configurazione del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class TrainingLogger:
    def __init__(self):
        self.log_dir = 'C:/Users/aldos/Desktop/hawk3/log_trainig_hawk'
        self.models_dir = os.path.join(self.log_dir, 'models')
        self.plots_dir = os.path.join(self.log_dir, 'plots')
        self.metrics_dir = os.path.join(self.log_dir, 'metrics')
        
        # Creazione struttura directory
        for directory in [self.log_dir, self.models_dir, self.plots_dir, self.metrics_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Configurazione logging avanzato
        self._setup_loggers()
        
        # Inizializzazione storage metriche
        self.metrics_history = defaultdict(list)
        self.training_start_time = datetime.now()

    def _setup_loggers(self):
        # Logger per codice
        self.code_logger = self._create_logger(
            'code_logger',
            os.path.join(self.log_dir, 'code_execution.log'),
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Logger per training
        self.training_logger = self._create_logger(
            'training_logger',
            os.path.join(self.log_dir, 'training_details.log'),
            '%(asctime)s - %(levelname)s - %(message)s'
        )

    def _create_logger(self, name, filepath, format_string):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(filepath)
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(console_handler)
        
        return logger

    def log_training_metrics(self, metrics, model_name, epoch=None):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Aggiornamento storico metriche
        for metric_name, value in metrics.items():
            self.metrics_history[f"{model_name}_{metric_name}"].append(value)
        
        # Conversione dei valori numpy in tipi Python nativi
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.integer):
                serializable_metrics[k] = int(v)
            elif isinstance(v, np.floating):
                serializable_metrics[k] = float(v)
            else:
                serializable_metrics[k] = v
        
        # Salvataggio metriche
        metrics_file = os.path.join(self.metrics_dir, f'metrics_{model_name}_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump({
                'model_name': model_name,
                'epoch': int(epoch) if epoch is not None else None,
                'metrics': serializable_metrics,
                'timestamp': timestamp
            }, f, indent=4)
        
        # Generazione grafici
        self._generate_training_plots(model_name, timestamp)
        
        # Log dettagliato
        self.training_logger.info(f"\nModel: {model_name} - Epoch: {epoch}\n" + 
                                "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    def _generate_training_plots(self, model_name, timestamp):
        # Plot metriche nel tempo
        plt.figure(figsize=(15, 8))
        for metric_name, values in self.metrics_history.items():
            if model_name in metric_name:
                plt.plot(values, label=metric_name)
        
        plt.title(f'Training Metrics Evolution - {model_name}')
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_file = os.path.join(self.plots_dir, f'training_evolution_{model_name}_{timestamp}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

    def log_code_execution(self, message, level='info'):
        getattr(self.code_logger, level)(message)

    def log_training_progress(self, message, metrics=None):
        self.training_logger.info(message)
        if metrics:
            self.training_logger.info("Current metrics:\n" + 
                                    "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))

    def save_training_summary(self):
        training_duration = datetime.now() - self.training_start_time
        summary = {
            'training_duration': str(training_duration),
            'final_metrics': {k: v[-1] for k, v in self.metrics_history.items()},
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        with open(os.path.join(self.log_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)


MODEL_DIR = 'C:/Users/aldos/Desktop/hawk3/hawk_ia'

def save_models(model, scaler, tfidf, feature_names, timestamp=None):
    # Crea directory se non esiste
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Aggiungi timestamp al nome del file per versioning
    timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Dizionario dei componenti da salvare
    model_components = {
        'model': model,
        'scaler': scaler,
        'tfidf': tfidf,
        'feature_names': feature_names,
        'metadata': {
            'saved_at': timestamp,
            'model_type': type(model).__name__,
            'features_count': len(feature_names)
        }
    }
    
    # Salva ogni componente con gestione errori
    for name, component in model_components.items():
        try:
            filepath = os.path.join(MODEL_DIR, f'{name}_{timestamp}.joblib')
            joblib.dump(component, filepath)
        except Exception as e:
            raise RuntimeError(f"Errore nel salvataggio di {name}: {str(e)}")
            
    # Salva riferimento all'ultima versione
    with open(os.path.join(MODEL_DIR, 'latest_version.txt'), 'w') as f:
        f.write(timestamp)

def load_models(version=None):
    if not version:
        # Carica l'ultima versione disponibile
        try:
            with open(os.path.join(MODEL_DIR, 'latest_version.txt'), 'r') as f:
                version = f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError("Nessuna versione del modello trovata")
    
    try:
        model = joblib.load(os.path.join(MODEL_DIR, f'model_{version}.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, f'scaler_{version}.joblib'))
        tfidf = joblib.load(os.path.join(MODEL_DIR, f'tfidf_{version}.joblib'))
        feature_names = joblib.load(os.path.join(MODEL_DIR, f'feature_names_{version}.joblib'))
        metadata = joblib.load(os.path.join(MODEL_DIR, f'metadata_{version}.joblib'))
        
        return model, scaler, tfidf, feature_names
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento dei modelli: {str(e)}")

def models_exist():
    try:
        with open(os.path.join(MODEL_DIR, 'latest_version.txt'), 'r') as f:
            version = f.read().strip()
        
        required_files = [
            f'model_{version}.joblib',
            f'scaler_{version}.joblib',
            f'tfidf_{version}.joblib',
            f'feature_names_{version}.joblib'
        ]
        
        return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files)
    except FileNotFoundError:
        return False

def get_model_versions():
    """Restituisce tutte le versioni disponibili dei modelli"""
    if not os.path.exists(MODEL_DIR):
        return []
    
    versions = []
    for file in os.listdir(MODEL_DIR):
        if file.startswith('model_') and file.endswith('.joblib'):
            versions.append(file.replace('model_', '').replace('.joblib', ''))
    
    return sorted(versions, reverse=True)

# Definizione dei percorsi dei file
file_storico = Path('C:/Users/aldos/Desktop/hawk3/storico.csv')
file_multifibra = Path('C:/Users/aldos/Desktop/hawk3/multifibra.csv')
file_indirizzi_pte = Path('C:/Users/aldos/Desktop/hawk3/indirizzi_pte.csv')

# Definizione delle keywords
keywords_negative = {"palificazione", "attraversamento stradale", "facciata", "due pezzi di scala", 
                     "più pezzi di scala", "canalina ostruita", "pozzetto", "pozzetti", "tubazione ostruita", "intercapedine", "MONOFIBRA ESTERNO"}
keywords_positive = {"quarto piano", "primo piano", "secondo piano", "terzo piano", "quinto piano", 
                     "piano 1", "piano 2", "piano 3", "piano 4", "piano 5"}
Keywords_positive_edificio = {'quarto piano', 'Quarto piano', 'primo piano', 'Primo piano', 'secondo piano', 'Secondo piano',
                              'terzo piano', 'Terzo piano', 'quinto piano', 'Quinto piano', 'piano 1', 'Piano 1', 'piano 2', 
                              'Piano 2', 'piano 3', 'Piano 3', 'piano 4', 'Piano 4', 'piano 5', 'Piano 5', 'piano 6', 'Piano 6'}
Keywords_negative_edificio = {"casa singola", "costruzione indipendente"}

def normalize_string(s):
    # Converti in maiuscolo
    s = s.upper()
    
    # Sostituisci l'apostrofo con niente (rimuovilo)
    s = s.replace("'", "")
    
    # Rimuovi gli accenti
    s = ''.join(c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn')
    
    # Rimuovi tutti i caratteri non alfanumerici eccetto spazi
    s = re.sub(r'[^A-Z0-9 ]', '', s)
    
    # Rimuovi spazi multipli
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def load_and_preprocess_data():
    print("Caricamento e preprocessamento dei dati...")

    # Caricamento dei dati
    df_storico = pd.read_csv(file_storico, low_memory=False)
    df_multifibra = pd.read_csv(file_multifibra, low_memory=False)
    df_indirizzi_pte = pd.read_csv(file_indirizzi_pte, low_memory=False)

    # Preprocessamento df_storico
    df_storico['STREET'] = df_storico['STREET'].astype(str).apply(normalize_string)
    df_storico['CITY'] = df_storico['CITY'].astype(str).apply(normalize_string)
    df_storico['DATA'] = pd.to_datetime(df_storico['DATA'], errors='coerce')
    df_storico['CAUSALE'] = df_storico['CAUSALE'].astype(str).str.strip()
    df_storico['CAUSALENAME'] = df_storico['CAUSALENAME'].astype(str).str.strip()
    df_storico['manodopera'] = df_storico['manodopera'].astype(str).str.upper().str.strip()
    df_storico['NOTE_TECNICO'] = df_storico['NOTE_TECNICO'].astype(str)
    df_storico['ACT_NARRATIVE'] = df_storico['ACT_NARRATIVE'].astype(str)

    # Converti le colonne categoriche in stringhe e le colonne numeriche in float per df_storico
    for col in df_storico.columns:
        if df_storico[col].dtype == 'object':
            df_storico[col] = df_storico[col].astype(str)
        elif df_storico[col].dtype in ['int64', 'float64']:
            df_storico[col] = df_storico[col].astype(float)

    # Converti la colonna 'DATA' in datetime (già fatto sopra, ma lo manteniamo per sicurezza)
    df_storico['DATA'] = pd.to_datetime(df_storico['DATA'], errors='coerce')

    # Ordina il DataFrame per 'STREET' e 'DATA'
    df_storico = df_storico.sort_values(['STREET', 'DATA'])

    # Crea la colonna 'MOS_count'
    df_storico['MOS_count'] = df_storico.groupby('STREET').cumcount().where(df_storico['manodopera'] == 'MOS', 0)

    # Crea la colonna 'COMPLWR_count'
    df_storico['COMPLWR_count'] = df_storico.groupby('STREET').cumcount().where(df_storico['CAUSALENAME'] == 'COMPLWR', 0)

    df_storico['keywords_negative'] = df_storico['NOTE_TECNICO'].apply(lambda x: any(keyword in str(x).lower() for keyword in keywords_negative))
    df_storico['keywords_positive'] = df_storico['NOTE_TECNICO'].apply(lambda x: any(keyword in str(x).lower() for keyword in keywords_positive))

    # Preprocessamento df_multifibra
    df_multifibra['STREET_MULTI'] = df_multifibra['STREET_MULTI'].astype(str).apply(normalize_string)
    df_multifibra['CITY_MULTI'] = df_multifibra['CITY_MULTI'].astype(str).apply(normalize_string)

    # Preprocessamento df_indirizzi_pte
    df_indirizzi_pte['STREET_PTE'] = df_indirizzi_pte['STREET_PTE'].astype(str).apply(normalize_string)
    df_indirizzi_pte['CITY_PTE'] = df_indirizzi_pte['CITY_PTE'].astype(str).apply(normalize_string)
    df_indirizzi_pte['TIPO_ROE'] = df_indirizzi_pte['TIPO_ROE'].astype(str).str.strip()
    df_indirizzi_pte['Apparato_UBICAZIONE'] = df_indirizzi_pte['Apparato_UBICAZIONE'].astype(str).str.strip()

    # Gestione dei valori mancanti
    df_indirizzi_pte['TIPO_ROE'] = df_indirizzi_pte['TIPO_ROE'].fillna('SCONOSCIUTO')
    df_indirizzi_pte['Apparato_UBICAZIONE'] = df_indirizzi_pte['Apparato_UBICAZIONE'].fillna('SCONOSCIUTO')

    # Rimuovi eventuali duplicati
    df_storico.drop_duplicates(inplace=True)
    df_multifibra.drop_duplicates(inplace=True)
    df_indirizzi_pte.drop_duplicates(inplace=True)

    # Resetta gli indici dopo tutte le modifiche
    df_storico.reset_index(drop=True, inplace=True)
    df_multifibra.reset_index(drop=True, inplace=True)
    df_indirizzi_pte.reset_index(drop=True, inplace=True)

    print(f"Dati storico caricati. Numero di righe: {len(df_storico)}")
    print(f"Dati multifibra caricati. Numero di righe: {len(df_multifibra)}")
    print(f"Dati indirizzi PTE caricati. Numero di righe: {len(df_indirizzi_pte)}")

    return df_storico, df_multifibra, df_indirizzi_pte

def serialize_prediction_result(prediction_result):
    """
    Serializza i risultati delle predizioni con supporto per voting multiplo
    e calibrazione dei pesi.
    """
    if isinstance(prediction_result, dict):
        serialized = {}
        for key, value in prediction_result.items():
            if isinstance(value, pd.Timestamp):
                serialized[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            elif key == 'predictions' and isinstance(value, list):
                serialized[key] = []
                total_weight = sum(pred.get('weight', 1.0) for pred in value)
                for pred in value:
                    weight = pred.get('weight', 1.0) / total_weight
                    confidence = pred.get('confidence', 0.5)
                    calibrated_confidence = (confidence * weight)
                    serialized[key].append({
                        'model': pred.get('model', 'unknown'),
                        'prediction': pred.get('prediction'),
                        'confidence': calibrated_confidence,
                        'weight': weight,
                        'rules_applied': pred.get('rules_applied', [])
                    })
            else:
                serialized[key] = serialize_prediction_result(value)
        return serialized
    elif isinstance(prediction_result, list):
        return [serialize_prediction_result(item) for item in prediction_result]
    return prediction_result

class FeedbackSystem:
    def __init__(self):
        self.feedback_store = {}
        self.rule_weights = defaultdict(lambda: 1.0)
    
    def store_feedback(self, prediction_id, actual_outcome, prediction_details):
        self.feedback_store[prediction_id] = {
            'actual': actual_outcome,
            'predicted': prediction_details['prediction'],
            'rules_applied': prediction_details['rules_applied']
        }
        self._update_rule_weights(prediction_details['rules_applied'], 
                                actual_outcome == prediction_details['prediction'])
    
    def _update_rule_weights(self, rules_applied, was_correct):
        adjustment = 0.1 if was_correct else -0.05
        for rule in rules_applied:
            self.rule_weights[rule] += adjustment
            self.rule_weights[rule] = max(0.1, min(2.0, self.rule_weights[rule]))

class RuleTracker:
    def __init__(self):
        self.applied_rules = []

    def track_rule(self, rule_name, score_impact, description, reason=None):
        """
        Traccia l'applicazione di una regola
        
        Parameters:
        - rule_name: nome della regola
        - score_impact: impatto sul punteggio 
        - description: descrizione della regola
        - reason: motivazione opzionale dell'applicazione della regola
        """
        rule_info = {
            'name': rule_name,
            'impact': score_impact,
            'description': description
        }
        
        if reason:
            rule_info['reason'] = reason
            
        self.applied_rules.append(rule_info)
        
    def get_rule_history(self):
        return pd.DataFrame(self.applied_rules)
    
    def clear_tracking(self):
        self.applied_rules = []

    def get_applied_rules(self):
        """
        Restituisce la lista delle regole applicate
        """
        return [
            {
                'name': rule['name'],
                'impact': rule['impact'], 
                'description': rule['description']
            }
            for rule in self.applied_rules
        ]

class BusinessRuleLearner:
    def __init__(self):
        self.rule_weights = defaultdict(lambda: 1.0)
        self.rule_history = []
        self.feedback_store = {}

    def learn_from_feedback(self, prediction_id, actual_outcome, prediction_details):
        """Apprende dalle regole di business applicate"""
        if prediction_id in self.feedback_store:
            feedback = self.feedback_store[prediction_id]
            was_correct = actual_outcome == feedback['predicted']
            
            # Aggiorna i pesi delle regole
            for rule in feedback['rules_applied']:
                self._update_rule_weight(rule, was_correct)
                
            # Traccia l'apprendimento
            self.rule_history.append({
                'prediction_id': prediction_id,
                'rules': feedback['rules_applied'],
                'outcome': was_correct,
                'weights': dict(self.rule_weights)
            })

    def _update_rule_weight(self, rule, was_correct):
        """Aggiorna il peso di una regola basato sul feedback"""
        adjustment = 0.1 if was_correct else -0.05
        self.rule_weights[rule] += adjustment
        # Mantieni i pesi in un range ragionevole
        self.rule_weights[rule] = max(0.1, min(2.0, self.rule_weights[rule]))

    def save_state(self, filepath):
        """Salva lo stato corrente su file JSON"""
        state = {
            'rule_weights': dict(self.rule_weights),
            'rule_history': self.rule_history
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load_state(self, filepath):
        """Carica lo stato da file JSON"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    state = json.load(f)
                    self.rule_weights = defaultdict(lambda: 1.0, state.get('rule_weights', {}))
                    self.rule_history = state.get('rule_history', [])
                return True
            return False
        except Exception as e:
            print(f"Creazione nuovo stato delle regole: {filepath}")
            self.save_state(filepath)
            return False

class PredictionLogger:
    def __init__(self):
        self.predictions = []
        self.timestamps = []
        
    def log_prediction(self, prediction, metadata):
        timestamp = datetime.now()
        self.predictions.append({
            'prediction': prediction,
            'metadata': metadata,
            'timestamp': timestamp
        })
        self.timestamps.append(timestamp)
        
    def get_predictions(self):
        return self.predictions
        
    def get_latest_prediction(self):
        return self.predictions[-1] if self.predictions else None
        
    def clear_logs(self):
        self.predictions = []
        self.timestamps = []

class ManodoperaRuleEngine:
    def __init__(self):
        # Inizializzazione logger esistente
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Inizializzazione RuleTracker
        self.rule_tracker = RuleTracker()
        
        # Inizializzazione componenti esistenti
        self.feedback_system = FeedbackSystem()
        self.rule_learner = BusinessRuleLearner()
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='Logloss',
            verbose=False
        )
        self.calibrator = CalibratedClassifierCV()
        self.performance_monitor = PerformanceMonitor()
        self.ensemble = ModelEnsemble()
        self.feedback_data = []
        self.is_fitted = False

    def learn_from_feedback(self, prediction_id, actual_outcome, prediction_details):
        if prediction_id in self.feedback_store:
            feedback = self.feedback_store[prediction_id]
            was_correct = actual_outcome == feedback['predicted']
            
            # Aggiornamento pesi con feedback loop
            self._update_weights_with_feedback(feedback['rules_applied'], was_correct)
            
            # Calibrazione predizioni
            self._calibrate_predictions(prediction_details)
            
            # Monitoraggio performance
            self.performance_monitor.track_prediction(
                prediction_details['prediction'],
                actual_outcome,
                prediction_details['confidence']
            )
            
            # Tracciamento storico
            self.rule_history.append({
                'prediction_id': prediction_id,
                'rules': feedback['rules_applied'],
                'outcome': was_correct,
                'weights': dict(self.rule_weights)
            })
            
    def _update_weights_with_feedback(self, rules, was_correct):
        """Sistema di feedback loop per aggiornamento pesi"""
        for rule in rules:
            current_performance = self.performance_monitor.get_rule_performance(rule)
            adjustment = self._calculate_adjustment(was_correct, current_performance)
            self.rule_weights[rule] = max(0.1, min(2.0, self.rule_weights[rule] + adjustment))
            
    def _calibrate_predictions(self, prediction_details):
        """Calibrazione delle predizioni"""
        if hasattr(self, 'calibration_data'):
            self.calibrator.fit(
                self.calibration_data['predictions'],
                self.calibration_data['actuals']
            )
    
    def update_score(self, score, points, reason, rule_name):
        """
        Aggiorna il punteggio e traccia la regola applicata
        
        Args:
            score (float): Punteggio corrente
            points (float): Punti da aggiungere/sottrarre
            reason (str): Motivazione dell'aggiornamento
            rule_name (str): Nome identificativo della regola
            
        Returns:
            float: Nuovo punteggio aggiornato
        """
        new_score = score + points
        
        # Traccia l'applicazione della regola
        self.rule_tracker.track_rule(
            rule_name=rule_name,
            score_impact=points,
            description=reason  # Aggiunto il parametro description mancante
        )
        
        # Log dettagliato dell'operazione
        self.logger.info(
            f"Regola '{rule_name}' applicata:\n"
            f"- Impatto: {points:.2f} punti\n"
            f"- Motivo: {reason}\n"
            f"- Punteggio aggiornato: {new_score:.2f}"
        )
        
        return new_score

    def get_rule_statistics(self):
        """
        Restituisce statistiche sull'applicazione delle regole
        """
        df_rules = self.rule_tracker.get_rule_history()
        if not df_rules.empty:
            stats = df_rules.groupby('name').agg({  
                'impact': ['count', 'mean', 'sum'],
                'description': ['first'] 
            })
            return stats
        return pd.DataFrame()

    def detect_anomaly(self, features):
        """
        Rileva anomalie e traccia il risultato
        """
        if not self.is_fitted:
            return False
            
        try:
            features_array = np.array(features).reshape(1, -1)
            scaled_features = self.scaler.transform(features_array)
            prediction = self.anomaly_detector.predict(scaled_features)
            anomaly_score = self.anomaly_detector.score_samples(scaled_features)
            
            if prediction[0] == -1:
                self.rule_tracker.track_rule(
                    rule_name="anomaly_detection",
                    score_impact=-10,
                    reason=f"Anomalia rilevata con score: {anomaly_score[0]:.3f}"
                )
                self.logger.warning(f"Rilevata anomalia con score: {anomaly_score[0]:.3f}")
            
            return prediction[0] == -1
            
        except Exception as e:
            self.logger.error(f"Errore durante il rilevamento anomalie: {str(e)}")
            return False

    def calculate_base_weight(self, date):
        days_since_intervention = (datetime.now() - pd.to_datetime(date)).days
        return 1 / (1 + days_since_intervention / 365)
    
    def save_learning_state(self):
        """
        Salva lo stato di apprendimento del sistema, includendo:
        - Pesi delle regole
        - Feedback accumulato
        - Statistiche di performance
        """
        learning_state = {
            'rule_weights': self.feedback_system.rule_weights,
            'feedback_data': self.feedback_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # Crea la directory se non esiste
        os.makedirs('model_state', exist_ok=True)
        
        # Salva lo stato in un file JSON
        with open('model_state/learning_state.json', 'w') as f:
            json.dump(learning_state, f, indent=4)
        
        self.logger.info("Stato di apprendimento salvato con successo")

    def fuzzy_membership(self, value, low, high):
        return max(0, min(1, (value - low) / (high - low)))
    
    def train_anomaly_detector(self, historical_data):
        # Verifica che i dati esistano e siano nel formato corretto
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        # Assicurati che le colonne necessarie esistano
        required_columns = ['MOS_count', 'COMPLWR_count']
        for col in required_columns:
            if col not in historical_data.columns:
                historical_data[col] = 0
        
        # Prepara i dati per l'addestramento
        features = historical_data[required_columns].values
        
        # Addestra l'anomaly detector
        self.scaler.fit(features)
        scaled_features = self.scaler.transform(features)
        self.anomaly_detector.fit(scaled_features)
        self.is_fitted = True
        
        return True

    def apply_rules(self, row, df_multifibra, df_indirizzi_pte):
        # Inizializzazione del sistema di apprendimento delle regole
        if not hasattr(self, 'rule_learner'):
            self.rule_learner = BusinessRuleLearner()
            
        # Inizializzazione esistente
        rule_tracker = RuleTracker()
        base_weight = self.calculate_base_weight(row['DATA'])
        score = 0
        prediction_id = str(uuid.uuid4())

        # Ottieni i pesi appresi delle regole
        learned_weights = self.rule_learner.rule_weights
        
        # Sistema di apprendimento adattivo per le regole
        def apply_learned_rule(rule_name, base_impact, condition):
            nonlocal score
            if condition:
                learned_weight = learned_weights[rule_name]
                adjusted_impact = base_impact * learned_weight * base_weight
                score = self.update_score(
                    score, 
                    adjusted_impact,
                    f"Regola {rule_name} applicata con peso appreso {learned_weight:.2f}",
                    rule_name
                )
                rule_tracker.track_rule(rule_name, adjusted_impact, f"Peso appreso: {learned_weight:.2f}")
                return True
            return False
        
        # Trova le informazioni PTE (codice esistente)
        pte_info = df_indirizzi_pte[(df_indirizzi_pte['CITY_PTE'] == row['CITY']) & 
                                    (df_indirizzi_pte['STREET_PTE'] == row['STREET'])]
        
        tipo_roe = pte_info['TIPO_ROE'].iloc[0] if not pte_info.empty else 'SCONOSCIUTO'
        apparato_ubicazione = pte_info['Apparato_UBICAZIONE'].iloc[0] if not pte_info.empty else 'SCONOSCIUTO'

        # Verifica la presenza di multifibra
        has_multifibra = not df_multifibra[(df_multifibra['CITY_MULTI'] == row['CITY']) & 
                                        (df_multifibra['STREET_MULTI'] == row['STREET'])].empty
        
        # Definizione delle causali da verificare
        causali_da_verificare = [
            'CREATION NON COMPLETATA', 'ROE NON ILLUMINATO', 'ROE SATURO', 
            'ROE PTE SATURO', 'ARMADIO F.O. SATURO', 'KO CANALINA DI EDIFICIO OSTRUITA',
            'KO CANALINA VERTICALE DI EDIFICIO OSTRUITA', 'KO CANALINA CLIENTE OSTRUITA',
            'KO CANALINA ORIZZONTALE DI EDIFICIO OSTRUITA', 'ATTESA PERMESSI',
            'KO INFRASTRUTTURA CLIENTE'
        ]

        # Regola 1: MOI senza precedenti MOS per COMPLWR
        apply_learned_rule(
            'rule_moi_no_mos',
            20 * self.fuzzy_membership(row['MOS_count'], 0, 5),
            row['manodopera'] == 'MOI' and row['MOS_count'] == 0 and row['CAUSALENAME'] == 'COMPLWR'
        )

        # Regola 2: Casa singola con COMPLWR
        apply_learned_rule(
            'rule_casa_singola',
            20 * self.fuzzy_membership(row['COMPLWR_count'], 0, 10),
            'casa singola' in str(row['ACT_NARRATIVE']).lower() and row['CAUSALENAME'] == 'COMPLWR'
        )

        # Regola 3: MOI con condizioni specifiche
        apply_learned_rule(
            'rule_moi_conditions',
            30,
            row['manodopera'] == 'MOI' and row['CAUSALENAME'] == 'COMPLWR' and 
            row['keywords_negative'] and not has_multifibra and 
            tipo_roe != 'ARMADIETTO' and apparato_ubicazione != 'ROE Interno'
        )

        # Regola 4: Keywords negative
        apply_learned_rule(
            'rule_negative_keywords',
            -10,
            row['keywords_negative']
        )

        # Regola 5: Keywords positive
        apply_learned_rule(
            'rule_positive_keywords',
            20,
            row['keywords_positive']
        )

        # Regola 6: MOS precedente per COMPLWR con supporto per singolo intervento
        mos_previous_activated = apply_learned_rule(
            'rule_mos_previous',
            70 * self.fuzzy_membership(max(1, row['COMPLWR_count']), 1, 15),
            row['manodopera'] == 'MOS' and row['CAUSALENAME'] == 'COMPLWR'
        )

        if mos_previous_activated:
            # Bonus aggiuntivo per primo intervento MOS riuscito
            if row['COMPLWR_count'] == 1:
                score += 90  # Punteggio maggiorato per primo intervento
            else:
                score += 80  # Punteggio standard per interventi successivi

        # Regola 7: Causale C31
        apply_learned_rule(
            'rule_c31',
            -10,
            row['CAUSALENAME'] == 'C31' and row['manodopera'] == 'MOS'
        )

        # Regola 8: Alto numero di interventi COMPLWR
        apply_learned_rule(
            'rule_high_complwr',
            20 * self.fuzzy_membership(row['COMPLWR_count'], 7, 15),
            row['manodopera'] == 'MOI' and row['COMPLWR_count'] >= 7
        )

        # Regola 9: Causali specifiche
        if row['CAUSALE'] in ['C42', 'N43', 'N44', 'N72']:
            if row['keywords_positive']:
                apply_learned_rule('rule_specific_causes_positive', 20, True)
            elif row['keywords_negative']:
                apply_learned_rule('rule_specific_causes_negative', -10, True)
            else:
                apply_learned_rule('rule_specific_causes_neutral', 
                                5 if row['manodopera'] == 'MOS' else -5, 
                                True)

        # Regola 10: Presenza di multifibra
        if apply_learned_rule('rule_multifibra', 35, has_multifibra):
            return 'MOS', 0.7, 90

        # Regola 11: ROE interno senza problemi di canalina
        if apply_learned_rule(
            'rule_roe_internal',
            45,
            tipo_roe == 'ARMADIETTO' and apparato_ubicazione == 'ROE Interno' and 
            'CANALINA' not in str(row['CAUSALE'])
        ):
            return 'MOS', 0.9, 90

        # Regola 12: Note tecnico
        if apply_learned_rule(
            'rule_note_tecnico',
            50,
            ('monofibra' in str(row['NOTE_TECNICO']).lower() and 'interno' in str(row['NOTE_TECNICO']).lower()) or
            any(multifibra in str(row['NOTE_TECNICO']).upper() for multifibra in 
                ['MULTIFIBRA:SI', 'MULTIFIBRA: SI', 'MULTIFIBRA :SI', 'MULTIFIBRA : SI'])
        ):
            return 'MOS', 0.9, 95

        # Regola 13: Causali critiche
        apply_learned_rule(
            'rule_critical_causes',
            15,
            any(row['CAUSALE'] == causale for causale in causali_da_verificare)
        )

        # Rilevamento anomalie
        features = [row['MOS_count'], row['COMPLWR_count']]
        if self.detect_anomaly(features):
            self.logger.warning(f"Rilevata anomalia per l'indirizzo: {row['STREET']}")
            score = self.update_score(score, -10 * base_weight, "Anomalia rilevata", "rule_anomaly")

        # Normalizza il punteggio finale in una scala da 0 a 100
        normalized_score = min(max(score, 0), 100)
        final_prediction = 'MOS' if normalized_score > 50 else 'MOI'
        confidence = normalized_score / 100

        # Aggiunta del feedback di apprendimento
        prediction_details = {
            'prediction': final_prediction,
            'rules_applied': [rule['name'] for rule in rule_tracker.applied_rules],
            'base_weight': base_weight,
            'normalized_score': normalized_score,
            'learned_weights': dict(learned_weights)
        }

        # Salvataggio del feedback per l'apprendimento
        self.feedback_system.store_feedback(prediction_id, None, prediction_details)
        
        # Aggiornamento dell'apprendimento delle regole
        self.rule_learner.learn_from_feedback(
            prediction_id,
            None,  # actual_outcome sarà fornito successivamente
            prediction_details
        )

        # Salvataggio dello stato di apprendimento
        self.save_learning_state()
        
        # Log delle performance di apprendimento
        self.logger.info(f"Pesi appresi delle regole: {dict(learned_weights)}")
        self.logger.info(f"Previsione finale per {row['STREET']}: {final_prediction} con punteggio {normalized_score:.2f}/100")
        
        return final_prediction, confidence, normalized_score
        
class ModelEnsemble:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(),
            'xgb': XGBClassifier(),
            'lgbm': LGBMClassifier()
        }
        self.weights = defaultdict(lambda: 1.0)

    def update_weights(self, model_performances):
        """Aggiorna i pesi dei modelli basati sulle performance"""
        total_performance = sum(model_performances.values())
        for model, perf in model_performances.items():
            self.weights[model] = perf / total_performance

class PerformanceMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.confidences = []
        self.rule_performances = defaultdict(list)

    def track_prediction(self, prediction, actual, confidence):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        self.confidences.append(confidence)

    def get_rule_performance(self, rule):
        """Calcola performance media della regola"""
        if rule in self.rule_performances:
            return np.mean(self.rule_performances[rule])
        return 0.5

def apply_manodopera_rules(interventi, rule_engine, df_multifibra, df_indirizzi_pte):
    predictions = []
    
    # Converti Series in DataFrame se necessario
    if isinstance(interventi, pd.Series):
        interventi = pd.DataFrame([interventi])
    
    for _, intervento in interventi.iterrows():
        # Applica le regole di business
        prediction, confidence, score = rule_engine.apply_rules(intervento, df_multifibra, df_indirizzi_pte)
        
        # Calcola bonus/penalità aggiuntive
        score = calcola_modificatori_punteggio(
            score, 
            intervento, 
            df_multifibra, 
            rule_engine
        )
        
        # Costruisci il dizionario del risultato
        prediction_dict = costruisci_risultato_predizione(
            prediction, confidence, score,
            intervento, df_multifibra,
            rule_engine.rule_tracker.get_applied_rules()
        )
        
        predictions.append(prediction_dict)

    # Seleziona la predizione migliore
    if predictions:
        final_prediction = max(predictions, key=lambda x: x['score'])
        rule_engine.logger.info(
            f"Previsione finale per {final_prediction['address']}: "
            f"{final_prediction['prediction']} con punteggio {final_prediction['score']:.2f}/100"
        )
        return (
            final_prediction['prediction'],
            final_prediction['confidence'],
            final_prediction['score']
        )
    
    return (None, 0, 0)

def calcola_modificatori_punteggio(score, intervento, df_multifibra, rule_engine):
    # Verifica multifibra
    if verifica_presenza_multifibra(intervento, df_multifibra):
        score = rule_engine.update_score(
            score=score,
            points=10,
            reason="Presenza di multifibra",
            rule_name="multifibra_bonus"
        )
    
    # Verifica keywords negative
    keywords_negative_trovate = trova_keywords_negative(intervento)
    if keywords_negative_trovate:
        score = rule_engine.update_score(
            score=score,
            points=-5 * len(keywords_negative_trovate),
            reason=f"Presenza di {len(keywords_negative_trovate)} keywords negative",
            rule_name="negative_keywords_penalty"
        )
    
    # Verifica linea pronta
    if verifica_linea_pronta(intervento):
        score = rule_engine.update_score(
            score=score,
            points=15,
            reason="Probabile presenza di linea pronta",
            rule_name="probable_line_bonus"
        )
    
    return max(0, min(100, score))

def costruisci_risultato_predizione(prediction, confidence, score, intervento, df_multifibra, applied_rules):
    return {
        'prediction': prediction,
        'confidence': confidence,
        'score': score,
        'address': intervento['STREET'],
        'city': intervento['CITY'],
        'has_multifibra': verifica_presenza_multifibra(intervento, df_multifibra),
        'negative_keywords': trova_keywords_negative(intervento),
        'probable_line': verifica_linea_pronta(intervento),
        'applied_rules': applied_rules
    }

def verifica_presenza_multifibra(intervento, df_multifibra):
    return ((df_multifibra['CITY_MULTI'] == intervento['CITY']) & 
            (df_multifibra['STREET_MULTI'] == intervento['STREET'])).any()

def trova_keywords_negative(intervento):
    note_tecnico = str(intervento.get('NOTE_TECNICO', '')).lower()
    return [kw for kw in keywords_negative if kw.lower() in note_tecnico]

def verifica_linea_pronta(intervento):
    return 'probabile presenza di linea pronta' in str(intervento.get('NOTE_TECNICO', '')).lower()

def extract_num_letter(address):
    match = re.match(r".*?(\d+)(?:[/\s]?([A-Za-z]))?$", str(address))
    if match:
        number = int(match.group(1))
        letter = match.group(2)
        return number, letter
    return None, None

def find_similar_addresses(address, city, df_storico):
    # Normalizza l'input
    city = normalize_string(city)
    address = normalize_string(address)

    # Estrai strada, numero e lettera dall'indirizzo
    match = re.match(r"(.*?)\s*(\d+)(?:[/\s]?([A-Za-z]))?$", address)
    if not match:
        return []

    street, number, letter = match.groups()
    number = int(number)
    
    # Filtra per città e strada esatte
    exact_matches = df_storico[(df_storico['CITY'] == city) & 
                               (df_storico['STREET'].str.startswith(street))]
    
    if exact_matches.empty:
        return []

    similar_addresses = []
    
    # Se c'è una lettera, cerca prima variazioni della lettera
    if letter:
        for _, row in exact_matches.iterrows():
            num, let = extract_num_letter(row['STREET'])
            if num is not None and num == number and let and let != letter:
                similar_addresses.append(row['STREET'])
    
    # Se non abbiamo abbastanza risultati, cerca numeri civici vicini
    if len(similar_addresses) < 10:
        for _, row in exact_matches.iterrows():
            num, _ = extract_num_letter(row['STREET'])
            if num is not None and number - 5 <= num <= number + 5 and num != number:
                similar_addresses.append(row['STREET'])
    
    # Rimuovi eventuali duplicati e limita a 10 risultati
    similar_addresses = list(dict.fromkeys(similar_addresses))[:10]
    
    return similar_addresses

# Preparazione delle colonne addizionali
def calculate_additional_columns(df_storico):
    df_storico.loc[:, 'COMPLWR_count'] = (df_storico['CAUSALENAME'] == 'COMPLWR').cumsum()
    df_storico.loc[:, 'keywords_negative'] = df_storico['NOTE_TECNICO'].apply(lambda x: any(keyword in str(x).lower() for keyword in keywords_negative))
    df_storico.loc[:, 'keywords_positive'] = df_storico['NOTE_TECNICO'].apply(lambda x: any(keyword in str(x).lower() for keyword in keywords_positive))
    return df_storico

def prepare_features(df_storico, df_indirizzi_pte, df_multifibra, single_address=False, tfidf=None, feature_names=None):
    rule_engine = ManodoperaRuleEngine()
    
    # Preparazione features base
    df = df_storico.copy()
    df = df.reset_index(drop=True)
    
    # Definizione delle helper functions
    def check_multifibra(row):
        has_multi = not df_multifibra[
            (df_multifibra['CITY_MULTI'] == row['CITY']) & 
            (df_multifibra['STREET_MULTI'] == row['STREET'])
        ].empty
        if has_multi:
            rule_engine.rule_tracker.track_rule(
                "multifibra_check", 1.0, 
                f"Multifibra presente in {row['CITY']}, {row['STREET']}"
            )
        return has_multi

    def check_roe_interno(row):
        has_roe = not df_indirizzi_pte[
            (df_indirizzi_pte['CITY_PTE'] == row['CITY']) &
            (df_indirizzi_pte['STREET_PTE'] == row['STREET']) &
            (df_indirizzi_pte['Apparato_UBICAZIONE'] == 'ROE Interno')
        ].empty
        if has_roe:
            rule_engine.rule_tracker.track_rule(
                "roe_interno_check", 1.0,
                f"ROE interno presente in {row['CITY']}, {row['STREET']}"
            )
        return int(has_roe)

    def check_causali_critiche(causale):
        causali_critiche = {
            'CREATION NON COMPLETATA', 'ROE NON ILLUMINATO', 'ROE SATURO',
            'ROE PTE SATURO', 'ARMADIO F.O. SATURO', 'KO CANALINA DI EDIFICIO OSTRUITA',
            'KO CANALINA VERTICALE DI EDIFICIO OSTRUITA', 'KO CANALINA CLIENTE OSTRUITA',
            'KO CANALINA ORIZZONTALE DI EDIFICIO OSTRUITA', 'ATTESA PERMESSI',
            'KO INFRASTRUTTURA CLIENTE'
        }
        if pd.isna(causale):
            return 0
        is_critica = causale in causali_critiche
        if is_critica:
            rule_engine.rule_tracker.track_rule(
                "causale_critica", -1.0,
                f"Causale critica rilevata: {causale}"
            )
        return int(is_critica)

    # Applicazione delle regole con tutti i parametri necessari
    results = df.apply(
        lambda row: pd.Series(
            apply_manodopera_rules(row, rule_engine, df_multifibra, df_indirizzi_pte)
        ),
        axis=1
    )
    
    # Assegnazione dei risultati e calcolo features
    df['rule_score'] = results.apply(lambda x: x[2])
    df['has_multifibra'] = df.apply(check_multifibra, axis=1)
    df['has_roe_interno'] = df.apply(check_roe_interno, axis=1)
    df['has_keywords_positive'] = df['keywords_positive'].fillna(False).astype(int)
    df['has_keywords_negative'] = df['keywords_negative'].fillna(False).astype(int)
    df['has_causali_critiche'] = df['CAUSALE'].fillna('').apply(check_causali_critiche)

    # Features numeriche e categoriche
    numeric_features = [
        'COMPLWR_count', 'MOS_count', 'rule_score',
        'has_multifibra', 'has_roe_interno',
        'has_keywords_positive', 'has_keywords_negative',
        'has_causali_critiche'
    ]
    categorical_features = ['CAUSALENAME', 'CAUSALE']

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Features testuali
    if tfidf is None:
        tfidf = TfidfVectorizer(max_features=1000)
        text_features = tfidf.fit_transform(df['NOTE_TECNICO'].fillna(''))
    else:
        text_features = tfidf.transform(df['NOTE_TECNICO'].fillna(''))

    # Combinazione features
    X = pd.concat([
        df_encoded[numeric_features],
        df_encoded[[col for col in df_encoded.columns if col.startswith(tuple(categorical_features))]], 
        pd.DataFrame(text_features.toarray(), columns=[f'tfidf_{i}' for i in range(text_features.shape[1])])
    ], axis=1)

    # Gestione feature_names
    if feature_names is not None:
        missing_cols = set(feature_names) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[feature_names]

    # Pulizia e conversione tipi
    X.columns = X.columns.astype(str)
    X = X.replace(['UNKNOWN', np.inf, -np.inf], 0)
    X = X.fillna(0).astype(float)

    # Target
    y = df['manodopera'].map({'MOI': 0, 'MOS': 1})
    if not single_address:
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

    # Pesi features
    feature_weights = pd.Series(1.0, index=X.columns)
    rule_based_features = [col for col in X.columns if col in numeric_features]
    feature_weights[rule_based_features] = 2.0

    rule_engine.logger.info(f"Features generate: {X.shape[1]}, Distribuzione target: MOI={sum(y==0)}, MOS={sum(y==1)}")
    
    return X, y, tfidf, feature_weights

def train_models(X, y, n_epochs=10):
    logger = TrainingLogger()
    rule_tracker = RuleTracker()
    business_rule_learner = BusinessRuleLearner()
    logger.log_code_execution("Avvio addestramento modelli")
    
    try:
        # Conversione dati in float32 per ottimizzare la memoria
        X = X.astype(np.float32)
        X.columns = X.columns.astype(str)
        
        # Pulizia dati
        mask = ~X.isna().any(axis=1) & ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Log distribuzione classi iniziale
        logger.log_training_progress(f"Distribuzione classi iniziale: {np.bincount(y)}")
        
        rule_tracker.track_rule(
            "data_preparation",
            1.0,
            f"Dati preparati con dimensioni: {X.shape}"
        )
        
        if len(X) < 100:
            rule_tracker.track_rule(
                "dataset_check",
                -1.0,
                "Dataset insufficiente per addestramento"
            )
            return None, None

        # Split stratificato per mantenere la distribuzione delle classi
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling con float32
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = X_train_scaled.astype(np.float32)
        X_test_scaled = X_test_scaled.astype(np.float32)
        
        # SMOTE con ratio personalizzato per bilanciamento moderato
        sampling_strategy = 0.8  # Rapporto tra classe minoritaria e maggioritaria
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        X_train_resampled = X_train_resampled.astype(np.float32)
        
        logger.log_training_progress(f"Distribuzione classi dopo SMOTE: {np.bincount(y_train_resampled)}")
        
        # Modelli con configurazioni anti-overfitting
        models = {
            'Random Forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    max_features='sqrt',
                    n_jobs=-1,
                    random_state=42,
                    class_weight='balanced',
                    bootstrap=True,
                    oob_score=True
                ),
                'param_grid': {
                    'n_estimators': [150, 200],
                    'max_depth': [6, 8, 10],
                    'min_samples_split': [10, 15]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=5,
                    random_strength=1,
                    bagging_temperature=1,
                    random_state=42,
                    verbose=False,
                    eval_metric='Logloss',
                    class_weights=[1, 2]  # Peso maggiore per la classe minoritaria
                ),
                'param_grid': {
                    'learning_rate': [0.03, 0.05],
                    'depth': [5, 6],
                    'l2_leaf_reg': [3, 5]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=25,
                    max_depth=8,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    is_unbalance=True,
                    scale_pos_weight=2
                ),
                'param_grid': {
                    'n_estimators': [150, 200],
                    'learning_rate': [0.03, 0.05],
                    'num_leaves': [20, 25]
                }
            }
        }

        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
        }

        calibrated_models = {}
        
        # Training con early stopping e validazione incrociata
        for epoch in range(n_epochs):
            for name, model_info in models.items():
                rule_tracker.track_rule(
                    f"model_training_{name}_epoch_{epoch}",
                    1.0,
                    f"Inizio addestramento modello {name} - Epoch {epoch+1}/{n_epochs}"
                )
                
                # Cross-validation con multiple metriche
                grid_search = GridSearchCV(
                    estimator=model_info['model'],
                    param_grid=model_info['param_grid'],
                    cv=5,  # Aumentato numero fold
                    scoring=scoring,
                    refit='f1',  # Ottimizza per F1-score
                    n_jobs=-1
                )
                
                # Pesi campione basati sulle regole di business
                sample_weights = np.ones(len(X_train_resampled))
                for idx, x in enumerate(X_train_resampled):
                    applied_rules = rule_tracker.get_applied_rules()
                    rule_weight = np.mean([business_rule_learner.rule_weights[rule] for rule in applied_rules])
                    sample_weights[idx] *= rule_weight
                
                grid_search.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
                
                # Calibrazione probabilità con cross-validation
                calibrated_model = CalibratedClassifierCV(
                    grid_search.best_estimator_,
                    cv=5,
                    method='isotonic'  # Calibrazione più flessibile
                )
                calibrated_model.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)
                
                # Valutazione e feedback con monitoraggio confidenza
                predictions = calibrated_model.predict(X_test_scaled)
                probabilities = calibrated_model.predict_proba(X_test_scaled)
                
                logger.log_training_progress(
                    f"\nModello {name} - Epoch {epoch+1}:\n"
                    f"Distribuzione predizioni: {np.bincount(predictions)}\n"
                    f"Media confidenza classe 0: {probabilities[:, 0].mean():.3f}\n"
                    f"Media confidenza classe 1: {probabilities[:, 1].mean():.3f}"
                )
                
                for idx, (pred, true) in enumerate(zip(predictions, y_test)):
                    prediction_id = f"{name}_epoch_{epoch}_pred_{idx}"
                    prediction_details = {
                        'prediction': pred,
                        'rules_applied': rule_tracker.get_applied_rules(),
                        'confidence': probabilities[idx][pred]
                    }
                    business_rule_learner.learn_from_feedback(prediction_id, true, prediction_details)
                
                calibrated_models[name] = calibrated_model

        # Ensemble con pesi dinamici
        model_weights = []
        for name, model in calibrated_models.items():
            predictions = model.predict(X_test_scaled)
            f1 = f1_score(y_test, predictions)
            model_weights.append(f1)
        
        # Normalizza i pesi
        model_weights = np.array(model_weights) / sum(model_weights)
        
        voting_model = VotingClassifier(
            estimators=[(name, model) for name, model in calibrated_models.items()],
            voting='soft',
            weights=model_weights
        )
        voting_model.fit(X_train_resampled, y_train_resampled)
        
        # Valutazione finale
        y_pred = voting_model.predict(X_test_scaled)
        y_prob = voting_model.predict_proba(X_test_scaled)
        
        final_metrics = classification_report(y_test, y_pred, output_dict=True)
        
        logger.log_training_progress(
            f"\nRisultati finali:\n"
            f"Distribuzione predizioni: {np.bincount(y_pred)}\n"
            f"Media confidenza classe 0: {y_prob[:, 0].mean():.3f}\n"
            f"Media confidenza classe 1: {y_prob[:, 1].mean():.3f}\n"
            f"F1 Score: {final_metrics['weighted avg']['f1-score']:.3f}"
        )
        
        rule_tracker.track_rule(
            "final_evaluation",
            1.0,
            f"F1 Score: {final_metrics['weighted avg']['f1-score']:.3f}"
        )
        
        # Salva stato finale
        business_rule_learner.save_state('model_state/business_rules.json')
        
        return voting_model, scaler
        
    except Exception as e:
        rule_tracker.track_rule(
            "training_error",
            -1.0,
            f"Errore durante l'addestramento: {str(e)}"
        )
        logger.log_code_execution(f"Errore in train_models: {str(e)}")
        return None, None

def generate_prediction(idx, row, X, model, rule_engine, df_multifibra, df_indirizzi_pte):
    # Get rule engine result
    rule_result = rule_engine.apply_rules(row, df_multifibra, df_indirizzi_pte)
    
    # Set default values if rule result is invalid
    if not isinstance(rule_result, tuple) or len(rule_result) != 3:
        rule_prediction = 'MOS'
        rule_confidence = 0.5
        rule_score = 50.0
    else:
        rule_prediction, rule_confidence, rule_score = rule_result
    
    # Handle missing index by using first row
    if idx not in X.index:
        X_pred = X.iloc[0:1]
    else:
        X_pred = X.loc[idx:idx]
    
    # Get model prediction
    try:
        model_proba = model.predict_proba(X_pred)[0]
        model_prediction = 'MOS' if model_proba[1] > 0.5 else 'MOI'
        model_confidence = float(max(model_proba))
    except:
        model_prediction = rule_prediction
        model_confidence = 0.5
    
    # Calculate combined score
    combined_score = (rule_score + (model_confidence * 100 if model_prediction == rule_prediction else 0)) / 2
    
    # Always return a valid prediction tuple
    return (model_prediction, model_confidence, combined_score, row)

def check_multifibra(df_multifibra, city, address, return_details=False):
    mask = ((df_multifibra['CITY_MULTI'] == city) & 
            (df_multifibra['STREET_MULTI'] == address))
    has_multi = mask.any()
    
    if return_details and has_multi:
        return has_multi, df_multifibra[mask]
    return has_multi

def analyze_keywords(note_tecnico):
    keywords_negative = ['palificazione', 'attraversamento stradale', 'facciata', 
                        'due pezzi di scala', 'più pezzi di scala', 'canalina ostruita', 
                        'pozzetto', 'pozzetti', 'tubazione ostruita', 'intercapedine', 
                        'MONOFIBRA ESTERNO']
    return [kw for kw in keywords_negative if kw.lower() in str(note_tecnico).lower()]

def find_problems(address_df):
    problematic_causals = [
        'CREATION NON COMPLETATA', 'ROE NON ILLUMINATO', 'ROE SATURO', 
        'ROE PTE SATURO', 'ARMADIO F.O. SATURO', 'KO CANALINA DI EDIFICIO OSTRUITA',
        'KO CANALINA VERTICALE DI EDIFICIO OSTRUITA', 'KO CANALINA CLIENTE OSTRUITA',
        'KO CANALINA ORIZZONTALE DI EDIFICIO OSTRUITA', 'ATTESA PERMESSI',
        'KO INFRASTRUTTURA CLIENTE'
    ]
    return address_df[address_df['CAUSALE'].isin(problematic_causals)]['CAUSALE'].unique().tolist()

def generate_model_predictions(model, X_scaled):
    model_predictions = {}
    for idx, estimator in enumerate(model.estimators_):
        model_name = f"Estimator_{idx}"
        X_pred = X_scaled.values
        proba = estimator.predict_proba(X_pred)[0]
        
        # Handle single class probability
        if len(proba) == 1:
            prediction = 'MOI'
            confidence = float(proba[0])
        else:
            prediction = 'MOS' if proba[1] > 0.5 else 'MOI'
            confidence = float(max(proba))
            
        model_predictions[model_name] = {
            "predizione": prediction,
            "confidenza": confidence
        }
    return model_predictions

def truncate_notes(notes, max_length=200):
    notes_str = str(notes)
    return notes_str[:max_length] + "..." if len(notes_str) > max_length else notes_str

def analyze_addresses(address_df, df_multifibra, rule_engine, df_indirizzi_pte):
    analyzed_addresses = []
    for addr in address_df['STREET'].unique():
        addr_data = address_df[address_df['STREET'] == addr]
        if addr_data.empty:
            continue
            
        complwr_addr_data = addr_data[addr_data['CAUSALENAME'] == 'COMPLWR']
        if complwr_addr_data.empty:
            continue
            
        row = complwr_addr_data.iloc[0]
        prediction, confidence, addr_manodopera_weight = apply_manodopera_rules(
            row, rule_engine, df_multifibra, df_indirizzi_pte
        )
        
        analyzed_addresses.append({
            "via": addr,
            "manodopera": prediction,
            "peso": float(abs(addr_manodopera_weight)),
            "causale": "COMPLWR",
            "keywords_negative": analyze_keywords(row['NOTE_TECNICO']),
            "presenza_multifibra": check_multifibra(df_multifibra, row['CITY'], row['STREET']),
            "probabile_linea_pronta": 'probabile presenza di linea pronta' in str(prediction)
        })
    
    return analyzed_addresses


def predict_manodopera(df, df_multifibra, df_indirizzi_pte, address, city, model, scaler, tfidf, feature_names, rule_engine, business_rule_learner):
    rule_tracker = RuleTracker()
    prediction_logger = PredictionLogger()
    
    # Inizializzazione modello e scaler
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        if not df.empty:
            X_train, y_train, _, _ = prepare_features(df, df_indirizzi_pte, df_multifibra, single_address=False)
            model.fit(X_train, y_train)
    
    if scaler is None:
        scaler = StandardScaler()
        if not df.empty:
            X_train = prepare_features(df, df_indirizzi_pte, df_multifibra, single_address=True)[0]
            feature_names = X_train.columns.tolist()
            scaler.fit(X_train)
    
    # Normalizzazione input
    city = city.strip().upper()
    address = address.strip().upper()
    
    # Gestione corrispondenze
    base_number = re.search(r'\d+', address)
    base_address = re.sub(r'(\d+).*', r'\1', address) if base_number else address
    
    exact_match = df[(df['CITY'] == city) & (df['STREET'] == address)]
    similar_match = df[(df['CITY'] == city) & (df['STREET'].str.startswith(base_address))]
    
    if not exact_match.empty:
        address_df = exact_match.copy()
    elif not similar_match.empty:
        address_df = similar_match.copy()
    else:
        similar_addresses = find_similar_addresses(address, city, df)
        if not similar_addresses:
            return {"error": "Indirizzo non trovato"}
        address_df = df[df['STREET'].isin(similar_addresses)].copy()
    
    if address_df.empty:
        return {"error": "Nessun dato disponibile"}
    
    # Preparazione features
    address_df = calculate_additional_columns(address_df)
    X, _, _, feature_weights = prepare_features(
        address_df, 
        df_indirizzi_pte, 
        df_multifibra, 
        single_address=True, 
        tfidf=tfidf, 
        feature_names=feature_names
    )
    
    # Preprocessing features
    X = prepare_features_for_prediction(X, feature_names, feature_weights, scaler)
    
    # Predizioni
    predictions = []
    for idx, row in address_df.iterrows():
        try:
            prediction_data = generate_prediction(
                idx, row, X, model, rule_engine,
                df_multifibra, df_indirizzi_pte
            )
            if prediction_data:
                predictions.append(prediction_data)
        except Exception as e:
            continue
    
    if not predictions:
        return {"error": "Non è stato possibile generare previsioni"}
    
    final_prediction, final_confidence, final_score, best_row = max(predictions, key=lambda x: x[2])
    
    # Analisi aggiuntive
    has_multifibra = check_multifibra(df_multifibra, city, address)
    negative_keywords_found = analyze_keywords(best_row['NOTE_TECNICO'])
    found_problems = find_problems(address_df)
    model_predictions = generate_model_predictions(model, X)
    
    # Costruzione risultato
    raw_result = {
        "indirizzo": address,
        "citta": city,
        "manodopera_consigliata": final_prediction,
        "confidenza": float(final_confidence),
        "punteggio_combinato": float(final_score),
        "intervento_determinante": {
            "data": str(best_row['DATA']),
            "causale": str(best_row['CAUSALENAME']),
            "note": truncate_notes(best_row['NOTE_TECNICO'])
        },
        "presenza_multifibra": bool(has_multifibra),
        "keywords_negative": negative_keywords_found,
        "problemi_riscontrati": found_problems,
        "ragionamento_modelli": model_predictions,
        "indirizzi_utilizzati": analyze_addresses(address_df, df_multifibra, rule_engine, df_indirizzi_pte)
    }
    
    return raw_result

def prepare_features_for_prediction(X, feature_names, feature_weights, scaler):
    if isinstance(X, pd.Series):
        X = X.to_frame().T
    
    for col in set(feature_names) - set(X.columns):
        X[col] = 0
    
    X = X[feature_names]
    X_weighted = X * feature_weights
    
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X_weighted),
        columns=feature_names,
        index=X_weighted.index
    )
    
    return pd.DataFrame(
        scaler.transform(X_imputed),
        columns=feature_names,
        index=X_imputed.index
    )


def get_statistics(df_storico, df_indirizzi_pte, df_multifibra, address=None, city=None, causale=None):
    if address and city:
        df_stats = df_storico[(df_storico['STREET'].str.upper() == address.upper()) & 
                              (df_storico['CITY'].str.upper() == city.upper())].copy()
    elif address:
        df_stats = df_storico[df_storico['STREET'].str.upper() == address.upper()].copy()
    elif causale:
        df_stats = df_storico[df_storico['CAUSALE'] == causale].copy()
    else:
        df_stats = df_storico.copy()
    
    if df_stats.empty:
        return {"error": "Nessuna statistica disponibile per questo indirizzo o causale."}
    
    return format_statistics(df_stats, df_indirizzi_pte, df_multifibra, address, city)

def format_statistics(df_stats, df_indirizzi_pte, df_multifibra, address, city):
    stats = {}
    
    stats["indirizzo"] = address
    stats["citta"] = city
    stats["totale_interventi"] = len(df_stats)
    
    complwr_count = df_stats['CAUSALENAME'].value_counts().get('COMPLWR', 0)
    stats["interventi_complwr"] = complwr_count
    stats["percentuale_complwr"] = (complwr_count / stats["totale_interventi"]) * 100 if stats["totale_interventi"] > 0 else 0
    
    top_causals = df_stats['CAUSALE'].value_counts().nlargest(20)
    stats["dettaglio_interventi"] = [
        (causale, count, (count / stats["totale_interventi"]) * 100)
        for causale, count in top_causals.items()
    ]
    
    if 'DATA' in df_stats.columns:
        df_stats['DATA'] = pd.to_datetime(df_stats['DATA'], errors='coerce')
        valid_dates = df_stats['DATA'].dropna()
        if not valid_dates.empty:
            stats["primo_intervento"] = valid_dates.min().strftime('%d/%m/%Y')
            stats["ultimo_intervento"] = valid_dates.max().strftime('%d/%m/%Y')
            if len(valid_dates) > 1:
                avg_time_between = (valid_dates.max() - valid_dates.min()) / (len(valid_dates) - 1)
                stats["tempo_medio_interventi"] = avg_time_between.days
    
    stats["presenza_multifibra"] = ((df_multifibra['CITY_MULTI'] == city) & 
                                    (df_multifibra['STREET_MULTI'] == address)).any()
    
    keywords_negative = ['palificazione', 'attraversamento stradale', 'facciata', 'due pezzi di scala', 
                     'più pezzi di scala', 'canalina ostruita', 'pozzetto', 'pozzetti', 'tubazione ostruita', 'intercapedine', 'MONOFIBRA ESTERNO']
    stats["keywords_negative"] = [kw for kw in keywords_negative if df_stats['NOTE_TECNICO'].str.contains(kw, case=False, na=False).any()]
    
    problematic_causals = ['CREATION NON COMPLETATA', 'ROE NON ILLUMINATO', 'ROE SATURO', 'ROE PTE SATURO', 
                           'ARMADIO F.O. SATURO', 'KO CANALINA DI EDIFICIO OSTRUITA', 'KO CANALINA VERTICALE DI EDIFICIO OSTRUITA',
                           'KO CANALINA CLIENTE OSTRUITA', 'KO CANALINA ORIZZONTALE DI EDIFICIO OSTRUITA', 'ATTESA PERMESSI', 
                           'KO INFRASTRUTTURA CLIENTE']
    stats["problemi_riscontrati"] = df_stats[df_stats['CAUSALE'].isin(problematic_causals)]['CAUSALE'].unique().tolist()
    
    stats["suggerimento_analisi"] = "Si consiglia di verificare che i problemi riscontrati in precedenza siano stati risolti."
    
    if not df_stats.empty:
        sample_address = df_stats.iloc[0]
        pte_info = df_indirizzi_pte[(df_indirizzi_pte['CITY_PTE'] == sample_address['CITY']) & 
                                    (df_indirizzi_pte['STREET_PTE'] == sample_address['STREET'])]
        
        if not pte_info.empty:
            roe_type = pte_info['TIPO_ROE'].dropna().unique().tolist()
            apparato_ubicazione = pte_info['Apparato_UBICAZIONE'].dropna().unique().tolist()
            stats["informazioni_pte"] = f"Tipo di ROE: {', '.join(roe_type)}. Ubicazione dell'apparato: {', '.join(apparato_ubicazione)}"
        else:
            stats["informazioni_pte"] = ""
    
    return stats

def main():
    # Inizializzazione componenti con nuove funzionalità
    logger = TrainingLogger()
    rule_tracker = RuleTracker()
    business_rule_learner = BusinessRuleLearner()
    logger.log_code_execution("Avvio esecuzione principale")
    
    try:
        # Caricamento dati con tracking avanzato
        logger.log_training_progress("Inizio caricamento dati")
        df_storico, df_multifibra, df_indirizzi_pte = load_and_preprocess_data()
        rule_tracker.track_rule("data_loading", 1.0, "Dati caricati con successo")
        
        # Inizializzazione componenti avanzati
        rule_engine = ManodoperaRuleEngine()
        performance_monitor = PerformanceMonitor()
        model_ensemble = ModelEnsemble()
        
        # Preparazione dati storici con metriche
        historical_data = df_storico[['MOS_count', 'COMPLWR_count']].copy()
        logger.log_training_metrics({
            'records_count': len(historical_data),
            'missing_values': historical_data.isnull().sum().sum()
        }, 'data_preparation')
        
        # Addestramento anomaly detector con feedback
        if not rule_engine.train_anomaly_detector(historical_data):
            rule_tracker.track_rule("anomaly_detector", -1.0, "Errore nell'addestramento")
            logger.log_code_execution("Errore nell'addestramento dell'anomaly detector", level='error')
            return
        
        # Verifica colonne con logging avanzato
        required_columns = ['CITY', 'STREET', 'CAUSALE', 'NOTE_TECNICO', 'COMPLWR_count', 'MOS_count']
        missing_columns = [col for col in required_columns if col not in df_storico.columns]
        if missing_columns:
            rule_tracker.track_rule("column_check", -1.0, f"Colonne mancanti: {missing_columns}")
            logger.log_code_execution(f"Colonne mancanti: {missing_columns}", level='error')
            return

        # Gestione modelli con ensemble
        if models_exist():
            logger.log_training_progress("Caricamento modelli esistenti")
            model, scaler, tfidf, feature_names = load_models()
            business_rule_learner.load_state('model_state/business_rules.json')
            rule_tracker.track_rule("model_loading", 1.0, "Modelli caricati con successo")
            logger.log_training_metrics({
                'model_loaded': True,
                'features_count': len(feature_names)
            }, 'model_loading')
        else:
            logger.log_training_progress("Addestramento nuovi modelli")
            X, y, tfidf, feature_weights = prepare_features(
                df_storico, 
                df_indirizzi_pte, 
                df_multifibra,
                single_address=False,
                tfidf=None,
                feature_names=None
            )
            
            X = X.fillna(0)
            
            # Addestramento con monitoraggio avanzato
            model, scaler = train_models(X, y, n_epochs=10)
            feature_names = X.columns.tolist()
            
            # Salvataggio con metriche
            save_models(model, scaler, tfidf, feature_names)
            logger.log_training_metrics({
                'features_trained': len(feature_names),
                'samples_used': len(X)
            }, 'model_training')
            rule_tracker.track_rule("model_training", 1.0, "Nuovi modelli addestrati con successo")

        # Addestramento anomaly detector con metriche
        historical_data = df_storico[['MOS_count', 'COMPLWR_count']]
        rule_engine.train_anomaly_detector(historical_data)
        logger.log_training_metrics({
            'anomaly_samples': len(historical_data)
        }, 'anomaly_detector')

        # Loop interattivo con tracking avanzato
        while True:
            logger.log_code_execution("Inizio nuova sessione di predizione")
            
            city = normalize_string(input("Inserisci la città (o 'q' per uscire): ").strip())
            if city.lower() == 'q':
                logger.save_training_summary()
                break
            
            address = normalize_string(input("Inserisci l'indirizzo: ").strip())
            rule_tracker.track_rule("user_input", 1.0, f"Input ricevuto: {city}, {address}")
            
            # Ricerca con performance tracking
            exact_match = df_storico[
                (df_storico['CITY'].str.upper() == city) & 
                (df_storico['STREET'].str.upper() == address)
            ]
            
            if not exact_match.empty:
                address_df = exact_match
                rule_tracker.track_rule("address_match", 2.0, f"Corrispondenza esatta trovata: {address}")
                used_address = address
                logger.log_training_metrics({
                    'exact_match_found': True
                }, 'address_search')
            else:
                similar_addresses = find_similar_addresses(address, city, df_storico)
                
                if not similar_addresses:
                    rule_tracker.track_rule("address_search", -1.0, "Nessun intervento trovato")
                    city_addresses = df_storico[df_storico['CITY'].str.upper() == city]['STREET'].unique()
                    logger.log_code_execution(f"Indirizzi disponibili per {city}:")
                    for addr in city_addresses[:10]:
                        logger.log_code_execution(f"- {addr}")
                    continue

                rule_tracker.track_rule("similar_addresses", 1.0, f"Trovati {len(similar_addresses)} indirizzi simili")
                similar_addresses = list(set(similar_addresses))
                for addr in similar_addresses:
                    logger.log_code_execution(f"- {addr}")

                address_df = df_storico[
                    (df_storico['STREET'].isin(similar_addresses)) & 
                    (df_storico['CITY'].str.upper() == city)
                ]
                used_address = address
            
            # Predizione con ensemble e monitoring
            prediction_result = predict_manodopera(
                address_df, 
                df_multifibra, 
                df_indirizzi_pte, 
                used_address, 
                city, 
                model, 
                scaler, 
                tfidf, 
                feature_names, 
                rule_engine,
                business_rule_learner
            )
            
            performance_monitor.track_prediction(
                prediction_result['prediction'],
                None,  # actual outcome non disponibile
                prediction_result['confidence']
            )
            
            rule_tracker.track_rule(
                "prediction", 
                1.0, 
                f"Predizione completata per {used_address}"
            )

            # Statistiche con metriche avanzate
            best_address = address_df.iloc[0]['STREET'] if not address_df.empty else used_address
            stats = get_statistics(df_storico, df_indirizzi_pte, df_multifibra, address=best_address, city=city)
            
            # Output risultati con serialize_prediction_result
            print("\nRisultato della predizione:")
            serialized_prediction = serialize_prediction_result(prediction_result)
            print(json.dumps(serialized_prediction, indent=2, ensure_ascii=False))
            
            # Tracking finale con metriche complete
            rule_tracker.track_rule(
                "session_complete", 
                1.0, 
                "Sessione di predizione completata con successo"
            )
            
            # Report completo
            tracking_report = rule_tracker.get_rule_history()
            logger.log_training_progress("\nReport di tracking delle regole:")
            logger.log_training_progress(tracking_report.to_string())
            
            # Salvataggio stato finale
            business_rule_learner.save_state('model_state/business_rules.json')
            logger.save_training_summary()

    except Exception as e:
        rule_tracker.track_rule("error", -1.0, f"Errore nell'esecuzione: {str(e)}")
        logger.log_code_execution(f"Errore nell'esecuzione principale: {str(e)}", level='error')
        raise e

if __name__ == "__main__":
    main()
