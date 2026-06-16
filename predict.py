import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
import requests
import json
import sys

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'models')

def get_groq_api_key():
    # 1. Try to read from environment variable
    key = os.environ.get("LLM_API_KEY") or os.environ.get("API_KEY") or os.environ.get("GROQ_API_KEY")
    if key:
        return key
        
    # 2. Try to read from .env file in project/backend or project root
    for path in [os.path.join(BASE, '.env'), os.path.join(os.path.dirname(BASE), '.env')]:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            clean_k = k.strip()
                            if clean_k in ['LLM_API_KEY', 'API_KEY', 'GROQ_API_KEY']:
                                return v.strip().strip('"').strip("'")
            except Exception:
                pass
    return ""

def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))
    svm = joblib.load(os.path.join(MODEL_DIR, 'svm.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    db = joblib.load(os.path.join(MODEL_DIR, 'dbscan.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl')) if os.path.exists(os.path.join(MODEL_DIR, 'label_encoder.pkl')) else None
    metadata = joblib.load(os.path.join(MODEL_DIR, 'metadata.pkl')) if os.path.exists(os.path.join(MODEL_DIR, 'metadata.pkl')) else None
    return rf, svm, scaler, db, le, metadata

def preprocess_df(df, metadata=None):
    if metadata is None:
        # Fallback to legacy structure
        cols = ['protocol','flow_duration','packet_count','bytes_sent','bytes_received','pps']
        X = df[cols].fillna(0).astype(float).values
        return X
        
    feature_cols = metadata['feature_cols']
    categorical_cols = metadata['categorical_cols']
    categorical_mappings = metadata['categorical_mappings']
    
    X_data = []
    for col in feature_cols:
        if col not in df.columns:
            # If the column is missing in prediction dataset, fill with 0s
            X_data.append(np.zeros(len(df)))
        else:
            series = df[col]
            if col in categorical_cols:
                mapping = categorical_mappings[col]
                series_str = series.fillna('missing').astype(str)
                # Map unseen values to 0
                encoded_vals = series_str.map(lambda x: mapping.get(x, 0)).values
                X_data.append(encoded_vals)
            else:
                X_data.append(series.fillna(0.0).astype(float).values)
                
    X = np.column_stack(X_data)
    return X

def check_anomaly(x, scaler, db):
    xs = scaler.transform([x])
    try:
        cores = db.components_
        idx, dist = pairwise_distances_argmin_min(xs, cores)
        return float(dist[0]) > db.eps
    except Exception:
        return False

def predict_with_groq(df):
    api_key = get_groq_api_key()
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Calculate DBSCAN anomalies to pass to Groq
    dbscan_anomalies = []
    try:
        rf, svm, scaler, db, le, metadata = load_models()
        X = preprocess_df(df, metadata)
        for x in X:
            anomaly = check_anomaly(x, scaler, db)
            dbscan_anomalies.append(bool(anomaly))
    except Exception as e:
        print(f"DBSCAN check failed for Groq input, defaulting to False: {e}", file=sys.stderr)
        dbscan_anomalies = [False] * len(df)
    
    # Exclude high cardinality or unnecessary columns to keep payload small
    exclude_keywords = ['timestamp', 'src_ip', 'dst_ip', 'id', 'uuid']
    cols_to_use = [c for c in df.columns if not any(kw in c.lower() for kw in exclude_keywords)]
    
    # Round floats to 2 decimal places to save tokens
    df_clean = df[cols_to_use].copy()
    for col in df_clean.select_dtypes(include=['float']).columns:
        df_clean[col] = df_clean[col].round(2)
        
    records = df_clean.to_dict(orient='records')
    # Inject dbscan_anomaly so Groq can apply rule 5 correctly
    for idx, rec in enumerate(records):
        rec['dbscan_anomaly'] = dbscan_anomalies[idx]
        
    results = []
    batch_size = 100
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        prompt = (
            "You are an expert Network Intrusion Detection System (IDS). Analyze these network traffic records.\n"
            "If the records contain 'Failed Login', 'Packets', 'Bytes', or 'Port' fields, classify them according to these rules:\n"
            "- DDoS (Label: 'DDoS', Severity: 'Critical', Anomaly: true): packets > 50000 OR bytes > 500000\n"
            "- Brute Force (Label: 'Brute Force', Severity: 'High', Anomaly: true): failed_login > 10\n"
            "- Port Scan (Label: 'Port Scan', Severity: 'High', Anomaly: true): port is 21 or 22\n"
            "- Malware (Label: 'Malware', Severity: 'Critical', Anomaly: true): port is 443 OR packets >= 1000\n"
            "- Normal (Label: 'Normal', Severity: 'Low', Anomaly: false): default fallback.\n\n"
            "Otherwise, if the records contain packet_loss, latency, congestion, or jitter, classify them strictly according to these rules (evaluated in priority order):\n"
            "1. DoS / DDoS (Label: '1', Severity: 'Critical', Anomaly: true): packet_loss > 30% OR latency > 500 ms OR congestion > 100\n"
            "2. Network Flooding (Label: '2', Severity: 'High', Anomaly: true): congestion > 70 AND throughput > 5\n"
            "3. Performance Anomaly (Label: '5', Severity: 'High', Anomaly: true): latency > 100 ms OR dbscan_anomaly is true\n"
            "4. Packet Drop Attack (Label: '3', Severity: 'Medium', Anomaly: true): packet_loss between 15% and 30% (inclusive)\n"
            "5. Jitter Attack (Label: '4', Severity: 'Medium', Anomaly: true): jitter > 5 ms\n"
            "6. Normal (Label: '0', Severity: 'Low', Anomaly: false): default fallback.\n\n"
            "Determine:\n"
            "1. 'attack': the class label string ('0', '1', '2', '3', '4', '5' or 'Normal', 'DDoS', 'Brute Force', 'Port Scan', 'Malware' strictly as returned by the matching rules)\n"
            "2. 'confidence': confidence percentage (0 to 100)\n"
            "3. 'anomaly': boolean (true if anomalous, false otherwise)\n"
            "4. 'severity': severity string ('Low', 'Medium', 'High', 'Critical' strictly matching the rule)\n\n"
            "Respond ONLY with a JSON object containing a single key 'predictions' mapping to a list of objects, one for each record in the exact input order. "
            "Do not include any text outside the JSON object.\n\n"
            f"Records:\n{batch}"
        )
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a precise network security classifier. Output raw JSON only matching the schema."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        preds = json.loads(content).get('predictions', [])
        
        # Align predictions length with batch
        if len(preds) < len(batch):
            for j in range(len(preds), len(batch)):
                preds.append({'attack': 'Normal', 'confidence': 50.0, 'anomaly': False, 'severity': 'Low'})
        elif len(preds) > len(batch):
            preds = preds[:len(batch)]
            
        results.extend(preds)
            
    return results
 
def generate_groq_summary(aggregates):
    api_key = get_groq_api_key()
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "You are an expert Network security dashboard assistant. Analyze the following aggregated IDS statistics and generate a security report:\n\n"
        f"Aggregates:\n{aggregates}\n\n"
        "Produce a JSON object with the following keys:\n"
        "1. 'summary': A concise summary (1-3 sentences) of the overall analysis, highlighting any attacks or anomalies detected. Do not mention Groq, Llama, AI, or machine learning algorithms in the summary.\n"
        "2. 'severity': The overall security risk level ('Normal', 'Medium', 'High', 'Critical').\n"
        "3. 'recommendations': A list of 2-4 actionable security recommendations based on the findings.\n"
        "4. 'key_metrics': A dictionary of 3-5 key-value pairs representing important metrics (e.g. 'Primary Threat', 'Total Anomalies', 'Confidence' formatted as a string with a '%' sign and 2 decimal places like '99.39%', etc.).\n\n"
        "Respond ONLY with the raw JSON object. Do not include any text outside the JSON object."
    )
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a network security reporter. Output raw JSON only matching the schema."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content']
    return json.loads(content)
 
def predict_df(df):
    try:
        print("Attempting prediction using Groq API...")
        results = predict_with_groq(df)
        print(f"Groq prediction succeeded. Received {len(results)} predictions.")
        return results, "Groq API (Llama 3.1 8B)"
    except Exception as e:
        print(f"Groq prediction failed, falling back to local models: {e}", file=sys.stderr)
        
        rf, svm, scaler, db, le, metadata = load_models()
        X = preprocess_df(df, metadata)
        Xs = scaler.transform(X)
 
        rf_pred = rf.predict(Xs)
        rf_proba = rf.predict_proba(Xs).max(axis=1) if hasattr(rf, 'predict_proba') else np.ones(len(Xs))
        svm_pred = svm.predict(Xs)
        svm_proba = svm.predict_proba(Xs).max(axis=1) if hasattr(svm, 'predict_proba') else np.ones(len(Xs))
 
        # Decode predictions back to original labels using target encoder 'le'
        if le is not None:
            rf_labels = le.inverse_transform(rf_pred)
            svm_labels = le.inverse_transform(svm_pred)
        else:
            rf_labels = rf_pred
            svm_labels = svm_pred
 
        results = []
        severity_map = {
            '0': 'Low',
            '1': 'Critical',
            '2': 'High',
            '3': 'Medium',
            '4': 'Medium',
            '5': 'High',
            'normal': 'Low',
            'ddos': 'Critical',
            'brute force': 'High',
            'port scan': 'High',
            'malware': 'Critical'
        }
        for i, x in enumerate(X):
            anomaly = check_anomaly(x, scaler, db)
            rf_label = str(rf_labels[i])
            svm_label = str(svm_labels[i])
            
            # decision logic
            if rf_label == svm_label:
                attack = rf_label
                confidence = float((rf_proba[i] + svm_proba[i]) / 2.0)
            else:
                # choose higher confidence
                if rf_proba[i] >= svm_proba[i]:
                    attack = rf_label
                    confidence = float(rf_proba[i])
                else:
                    attack = svm_label
                    confidence = float(svm_proba[i])
                    
            # Clean label to match key in severity_map
            clean_attack = attack.strip()
            # If the label is normal/benign, make it '0'
            if clean_attack.lower() in ['normal', 'benign', 'ok', 'safe', 'clean']:
                clean_attack = '0'
                
            severity = severity_map.get(clean_attack.lower(), 'Low')
            # Anomaly is True if DBSCAN flags it or it's a predicted attack (not '0' / 'normal')
            is_anomaly = bool(anomaly) or (clean_attack not in ['0', 'Normal'])
                
            results.append({
                'attack': clean_attack,
                'confidence': round(confidence * 100, 2),
                'anomaly': is_anomaly,
                'severity': severity
            })
        return results, "Local Models (Random Forest + SVM)"
