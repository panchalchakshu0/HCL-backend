"""Train models from a CSV dataset or synthetic fallback.
Saves: models/random_forest.pkl, models/svm.pkl, models/scaler.pkl, models/dbscan.pkl, models/label_encoder.pkl, models/metadata.pkl
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN

BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def find_label_column(df):
    target_names = ['label', 'class', 'anomaly', 'target', 'attack', 'attack_type', 'type', 'status', 'classification']
    # Check exact match case-insensitive (excluding false positives in networking)
    for col in df.columns:
        if col.lower() in target_names:
            if col.lower() in ['network target', 'video target']:
                continue
            return col
    # Check suffix match (excluding general target/type to avoid false positives)
    for col in df.columns:
        col_lower = col.lower()
        for tn in ['label', 'class', 'anomaly', 'attack', 'classification']:
            if col_lower.endswith('_' + tn) or col_lower.endswith(' ' + tn):
                return col
    # Check if there is only one string/object column
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) == 1:
        return cat_cols[0]
    # Default to last column
    return df.columns[-1]

def get_feature_columns(df, label_col=None):
    feature_cols = []
    for col in df.columns:
        if label_col and col == label_col:
            continue
        col_lower = col.lower()
        # Exclude IDs, timestamps, dates, and IP addresses
        exclude_keywords = ['id', 'uuid', 'timestamp', 'datetime', 'date', 'time', 'src_ip', 'dst_ip', 'ip_addr', 'ip']
        if any(kw in col_lower for kw in exclude_keywords):
            continue
        
        # Drop non-numeric columns with high cardinality (likely unique strings/hashes)
        if not pd.api.types.is_numeric_dtype(df[col]):
            nunique = df[col].nunique()
            if nunique > 50 and nunique > 0.3 * len(df):
                continue
                
        feature_cols.append(col)
    return feature_cols

def load_dataset(path_or_df=None):
    if isinstance(path_or_df, pd.DataFrame):
        return path_or_df
        
    if isinstance(path_or_df, str) and os.path.exists(path_or_df):
        return pd.read_csv(path_or_df)
        
    # Search in uploads directory automatically
    uploads_dir = os.path.join(BASE, 'uploads')
    if os.path.exists(uploads_dir):
        files = [f for f in os.listdir(uploads_dir) if f.endswith('.csv')]
        # Prioritize files with 'label' in their name
        labeled_files = [f for f in files if 'label' in f.lower()]
        if labeled_files:
            return pd.read_csv(os.path.join(uploads_dir, labeled_files[0]))
        # Fallback to other csv files that aren't test templates
        csv_files = [f for f in files if f not in ['sample_test.csv', 'test_upload.csv']]
        if csv_files:
            # Check if any csv has a label/anomaly column
            for f in csv_files:
                try:
                    temp_df = pd.read_csv(os.path.join(uploads_dir, f), nrows=5)
                    for col in temp_df.columns:
                        if col.lower() in ['label', 'class', 'anomaly', 'target', 'attack', 'type', 'status']:
                            return pd.read_csv(os.path.join(uploads_dir, f))
                except Exception:
                    continue
            # Otherwise, just return the first available csv file
            return pd.read_csv(os.path.join(uploads_dir, csv_files[0]))
            
    # Synthetic fallback
    n = 3000
    np.random.seed(1)
    df = pd.DataFrame({
        'src_ip': ['10.0.0.%d' % i for i in range(n)],
        'dst_ip': ['10.0.1.%d' % i for i in range(n)],
        'protocol': np.random.choice([6,17], n),
        'flow_duration': np.random.randint(1,10000,n),
        'packet_count': np.random.randint(1,1000,n),
        'bytes_sent': np.random.randint(40,1500,n),
        'bytes_received': np.random.randint(0,10000,n),
        'pps': np.random.rand(n)*100,
        'label': np.where(np.random.rand(n) < 0.85, 'Normal', np.random.choice(['DDoS','PortScan','BruteForce','Botnet'], n))
    })
    return df

def rule_based_labeling(df):
    # Check if this is the new packets/port dataset format
    has_new_features = any(col in df.columns for col in ['Packets', 'packets', 'Failed Login', 'failed_login', 'Port', 'port'])
    if has_new_features:
        labels = []
        for idx, row in df.iterrows():
            packets = float(row.get('Packets', row.get('packets', 0)))
            bytes_val = float(row.get('Bytes', row.get('bytes', 0)))
            failed_login = float(row.get('Failed Login', row.get('failed_login', 0)))
            port = float(row.get('Port', row.get('port', 80)))
            
            # Apply user's new network traffic classification rules:
            if packets > 50000 or bytes_val > 500000:
                labels.append('DDoS')
            elif failed_login > 10:
                labels.append('Brute Force')
            elif port == 21 or port == 22:
                labels.append('Port Scan')
            elif port == 443 or packets >= 1000:
                labels.append('Malware')
            else:
                labels.append('Normal')
        return np.array(labels)

    # Run a quick internal DBSCAN to identify outlier points (-1) for labeling
    label_col = find_label_column(df)
    feature_cols = get_feature_columns(df, label_col)
    X_data = []
    for col in feature_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            series_str = series.fillna('missing').astype(str)
            unique_vals = sorted(list(series_str.unique()))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            X_data.append(series_str.map(mapping).values)
        else:
            X_data.append(series.fillna(0.0).astype(float).values)
    X = np.column_stack(X_data)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    db = DBSCAN(eps=0.8, min_samples=10).fit(Xs)
    dbscan_labels = db.labels_
    
    labels = []
    for idx, row in df.iterrows():
        packet_loss = float(row.get('packet_loss', 0))
        latency = float(row.get('latency', 0))
        congestion = float(row.get('congestion', 0))
        throughput = float(row.get('throughput', 0))
        jitter = float(row.get('jitter', 0))
        is_dbscan_abnormal = (dbscan_labels[idx] == -1)
        
        # Apply user's cybersecurity classification rules in priority order:
        # 1. DoS / DDoS (1): packet_loss > 30% OR latency > 500 ms OR congestion > 100
        if packet_loss > 30.0 or latency > 500.0 or congestion > 100.0:
            labels.append(1)
        # 2. Network Flooding (2): congestion > 70 AND throughput > 5
        elif congestion > 70.0 and throughput > 5.0:
            labels.append(2)
        # 3. Performance Anomaly (5): latency > 100 ms OR abnormal DBSCAN point (-1)
        elif latency > 100.0 or is_dbscan_abnormal:
            labels.append(5)
        # 4. Packet Drop Attack (3): packet_loss between 15% - 30%
        elif 15.0 <= packet_loss <= 30.0:
            labels.append(3)
        # 5. Jitter Attack (4): jitter > 5 ms
        elif jitter > 5.0:
            labels.append(4)
        # 6. Normal (0): default (or packet_loss < 5 AND latency < 50 AND congestion < 30)
        else:
            labels.append(0)
            
    return np.array(labels)

def preprocess(df):
    label_col = find_label_column(df)
    feature_cols = get_feature_columns(df, label_col)
    
    # Use ground-truth labels if present and diverse, otherwise fall back to rules
    if label_col and label_col in df.columns and df[label_col].dropna().nunique() > 1:
        y_raw = df[label_col].fillna('Normal').astype(str).values
    else:
        y_raw = rule_based_labeling(df)
        
    le = LabelEncoder().fit(y_raw)
    y_enc = le.transform(y_raw)
    
    X_data = []
    categorical_cols = []
    categorical_mappings = {}
    
    for col in feature_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            categorical_cols.append(col)
            series_str = series.fillna('missing').astype(str)
            unique_vals = sorted(list(series_str.unique()))
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            categorical_mappings[col] = mapping
            X_data.append(series_str.map(mapping).values)
        else:
            X_data.append(series.fillna(0.0).astype(float).values)
            
    X = np.column_stack(X_data)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    metadata = {
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'categorical_mappings': categorical_mappings,
        'label_col': label_col
    }
    
    return Xs, y_enc, le, scaler, metadata

def train(path_or_df=None):
    df = load_dataset(path_or_df)
    Xs, y_raw, le, scaler, metadata = preprocess(df)

    rf = RandomForestClassifier(n_estimators=200)
    svm = SVC(kernel='rbf', probability=True)
    rf.fit(Xs, y_raw)
    svm.fit(Xs, y_raw)

    db = DBSCAN(eps=0.8, min_samples=10).fit(Xs)

    joblib.dump(rf, os.path.join(MODEL_DIR, 'random_forest.pkl'))
    joblib.dump(svm, os.path.join(MODEL_DIR, 'svm.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(db, os.path.join(MODEL_DIR, 'dbscan.pkl'))
    joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    joblib.dump(metadata, os.path.join(MODEL_DIR, 'metadata.pkl'))
    print('Saved models and metadata to', MODEL_DIR)

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    train(path)
