import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import pandas as pd
try:
    from .predict import predict_df
    from .database import SessionLocal
    from .models import Analysis
except ImportError:
    from predict import predict_df
    from database import SessionLocal
    from models import Analysis
from datetime import datetime

bp = Blueprint('api', __name__)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_friendly_attack_name_backend(attack):
    atk = str(attack).strip().lower()
    if atk in ['0', 'normal']: return 'Normal'
    if atk in ['1', 'dos', 'ddos'] or 'dos' in atk or 'ddos' in atk: return 'DoS / DDoS'
    if atk in ['2'] or 'flooding' in atk: return 'Network Flooding'
    if atk in ['3'] or 'packet drop' in atk or 'drop' in atk: return 'Packet Drop Attack'
    if atk in ['4'] or 'jitter' in atk: return 'Jitter Attack'
    if atk in ['5'] or 'performance' in atk or 'anomaly' in atk: return 'Performance Anomaly'
    if 'brute' in atk or 'force' in atk: return 'Brute Force'
    if 'scan' in atk or 'port' in atk: return 'Port Scan'
    if 'malware' in atk: return 'Malware'
    return attack.capitalize()

@bp.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'no file'}), 400
    filename = secure_filename(f.filename)
    path = os.path.join(UPLOAD_DIR, filename)
    f.save(path)
    return jsonify({'filename': filename}), 200

@bp.route('/predict', methods=['POST'])
def predict():
    # accept filename or raw csv in body
    if 'file' in request.files:
        f = request.files['file']
        df = pd.read_csv(f)
        filename = None
    else:
        data = request.get_json()
        filename = data.get('filename')
        if filename:
            path = os.path.join(os.path.dirname(__file__), 'uploads', filename)
            if not os.path.exists(path):
                return jsonify({'error': 'file not found'}), 404
            df = pd.read_csv(path)
        else:
            return jsonify({'error': 'no file provided'}), 400

    # Automatically trigger retraining if the uploaded dataset is labeled
    try:
        # Check if the dataset is labeled using a clean, high-confidence check
        target_names = ['label', 'class', 'anomaly', 'target', 'attack', 'attack_type', 'type', 'status', 'classification']
        has_label = False
        for col in df.columns:
            if col.lower() in target_names and col.lower() not in ['network target', 'video target']:
                has_label = True
                break
            for tn in ['label', 'class', 'anomaly', 'attack', 'classification']:
                if col.lower().endswith('_' + tn) or col.lower().endswith(' ' + tn):
                    has_label = True
                    break
            if has_label:
                break
                
        if has_label:
            try:
                from .train_model import train
            except ImportError:
                from train_model import train
            train(df)
    except Exception as e:
        current_app.logger.warning(f"Automatic retraining skipped: {e}")

    results, model_used = predict_df(df)
    # aggregate summary
    attacks = [r['attack'] for r in results]
    import collections
    counts = collections.Counter(attacks)
    
    # Prioritize reporting malicious attacks over normal traffic in the summary
    normal_labels = ['normal', '0', 'benign', 'ok', 'safe', 'clean']
    malicious_attacks = [a for a in attacks if a.lower() not in normal_labels]
    
    if malicious_attacks:
        malicious_counts = collections.Counter(malicious_attacks)
        most_attack = malicious_counts.most_common(1)[0][0]
    else:
        most_attack = counts.most_common(1)[0][0]
        
    avg_conf = sum(r['confidence'] for r in results) / len(results)
    anomalies = sum(1 for r in results if r['anomaly'])
    
    is_attack = most_attack.lower() not in normal_labels
    
    # Aggregates for dynamic reporting
    aggregates = {
        "total_records": len(results),
        "anomaly_count": anomalies,
        "primary_attack": most_attack,
        "has_attack": is_attack,
        "avg_confidence": f"{avg_conf:.2f}%",
        "attack_counts": dict(counts)
    }
    
    def generate_local_summary(aggr):
        sev = 'Critical' if aggr['anomaly_count'] > 0 and aggr['has_attack'] else ('High' if aggr['has_attack'] else 'Low')
        recs = [
            "Monitor the source network traffic for any repeating patterns.",
            "Check the anomalous packets in the uploaded traffic logs."
        ]
        if aggr['has_attack']:
            recs.append(f"Investigate firewall rules for the primary threat: {aggr['primary_attack']}.")
        metrics = {
            "Primary Threat": aggr['primary_attack'],
            "Confidence": aggr['avg_confidence'],
            "Anomalies Found": str(aggr['anomaly_count']),
            "Total Records": str(aggr['total_records'])
        }
        summ = f"Analysis completed using local classifiers. The most common predicted traffic classification is '{aggr['primary_attack']}' with an average confidence of {aggr['avg_confidence']}."
        return {
            "summary": summ,
            "severity": sev,
            "recommendations": recs,
            "key_metrics": metrics
        }
        
    try:
        try:
            from .predict import generate_groq_summary
        except ImportError:
            from predict import generate_groq_summary
        final = generate_groq_summary(aggregates)
        final['model_used'] = "Enterprise Detection System"
    except Exception as e:
        current_app.logger.warning(f"Groq summary generation failed, falling back to local summary: {e}")
        final = generate_local_summary(aggregates)
        final['model_used'] = "Enterprise Detection System"

    # Sanitize recommendations to always be a list of strings
    recs = final.get('recommendations', [])
    clean_recs = []
    if isinstance(recs, dict):
        try:
            sorted_keys = sorted(recs.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
            clean_recs = [str(recs[k]) for k in sorted_keys]
        except Exception:
            clean_recs = [str(v) for v in recs.values()]
    elif isinstance(recs, list):
        for item in recs:
            if isinstance(item, dict):
                val = next(iter(item.values()), "")
                clean_recs.append(str(val))
            else:
                clean_recs.append(str(item))
    else:
        clean_recs = []
    final['recommendations'] = clean_recs

    # Format the confidence metric in final key_metrics if it exists as float
    if 'key_metrics' in final and 'Confidence' in final['key_metrics']:
        try:
            conf_val = final['key_metrics']['Confidence']
            if isinstance(conf_val, (int, float)):
                final['key_metrics']['Confidence'] = f"{conf_val:.2f}%"
            elif isinstance(conf_val, str) and not conf_val.endswith('%'):
                final['key_metrics']['Confidence'] = f"{float(conf_val):.2f}%"
        except Exception:
            pass

    # Sanitize key_metrics to avoid nested dictionaries or arrays
    if 'key_metrics' in final and isinstance(final['key_metrics'], dict):
        clean_metrics = {}
        for k, v in final['key_metrics'].items():
            if isinstance(v, dict):
                sub_items = []
                for sub_k, sub_v in v.items():
                    friendly_sub_k = get_friendly_attack_name_backend(sub_k)
                    sub_items.append(f"{friendly_sub_k}: {sub_v}")
                clean_metrics[k] = ", ".join(sub_items)
            elif isinstance(v, list):
                clean_metrics[k] = ", ".join(str(item) for item in v)
            else:
                clean_metrics[k] = str(v)
        final['key_metrics'] = clean_metrics

    # persist analysis
    db = SessionLocal()
    attack_type = final.get('key_metrics', {}).get('Primary Threat', most_attack)
    a = Analysis(filename=filename or '', attack_type=attack_type, confidence=float(avg_conf), severity=final.get('severity', 'Low'), anomaly=(anomalies>0), total_records=len(results))
    db.add(a)
    db.commit()
    db.close()

    # Combine original record features with prediction results
    combined_records = []
    for idx, r in enumerate(results):
        row_dict = df.iloc[idx].fillna('').to_dict()
        row_dict['predicted_attack'] = r['attack']
        row_dict['predicted_confidence'] = f"{r['confidence']:.2f}%" if isinstance(r['confidence'], (int, float)) else str(r['confidence'])
        row_dict['predicted_anomaly'] = r['anomaly']
        row_dict['predicted_severity'] = r['severity']
        combined_records.append(row_dict)

    return jsonify({'final': final, 'per_record': combined_records}), 200

@bp.route('/history', methods=['GET'])
def history():
    db = SessionLocal()
    rows = db.query(Analysis).order_by(Analysis.created_at.desc()).all()
    out = []
    for r in rows:
        out.append({
            'id': r.id,
            'filename': r.filename,
            'attack_type': r.attack_type,
            'confidence': r.confidence,
            'severity': r.severity,
            'anomaly': r.anomaly,
            'total_records': r.total_records,
            'created_at': r.created_at.isoformat()
        })
    db.close()
    return jsonify(out)

@bp.route('/stats', methods=['GET'])
def stats():
    db = SessionLocal()
    total = db.query(Analysis).count()
    attacks = db.query(Analysis).filter(Analysis.severity != 'Low').count()
    anomalies = db.query(Analysis).filter(Analysis.anomaly == True).count()
    db.close()
    return jsonify({'total_analyses': total, 'attacks': attacks, 'anomalies': anomalies})
