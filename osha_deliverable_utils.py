"""Cálculo reproducible del primer entregable; no modifica los datos originales.

Las cachés guardan ajustes ya completados para no repetirlos al ejecutar el libro.
No hay selección de hiperparámetros, variables, particiones ni modelos.
"""
from pathlib import Path
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
                            f1_score, roc_auc_score, average_precision_score, confusion_matrix,
                            classification_report, log_loss)
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

SEED = 42
LABELS = [1, 2, 3, 4]
RAW_FEATURES = ['state','naics_code','establishment_type','soc_code','type_of_incident',
                'time_unknown','time_started_work','time_of_incident']
FEATURES = RAW_FEATURES[:6]+['time_started_work_hour','time_of_incident_hour']
PARAMS = dict(solver='newton-cg', max_iter=1000, class_weight='balanced', random_state=SEED)

def prepare_features(X):
    """Las mismas ocho variables del cribado, sin estadísticas aprendidas."""
    d = X.loc[:,RAW_FEATURES].copy().replace(r'^\s*$', np.nan, regex=True)
    out = d[RAW_FEATURES[:6]].copy()
    out['soc_code'] = out.soc_code.replace({'9999':np.nan,'0000':np.nan})
    for c in RAW_FEATURES[6:]:
        h = pd.to_numeric(d[c].str.extract(r'^(\d{1,2}):',expand=False),errors='coerce')
        out[c+'_hour'] = h.where(h.between(0,23)).map(lambda v: str(int(v)) if pd.notna(v) else np.nan)
    out.loc[d.time_unknown.eq('1'),'time_of_incident_hour'] = np.nan
    return out

def make_pipeline():
    return Pipeline([
        ('features',FunctionTransformer(prepare_features,validate=False)),
        ('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),
        ('onehot',OneHotEncoder(handle_unknown='ignore')),
        ('classifier',LogisticRegression(**PARAMS))])

def make_dummy():
    # No requiere transformar X: ignora todas las características.
    return Pipeline([('classifier',DummyClassifier(strategy='prior',random_state=SEED))])

def metrics(y,p):
    pred = np.asarray(LABELS)[p.argmax(axis=1)]
    precision, recall, f1, _ = precision_recall_fscore_support(y,pred,average='weighted',zero_division=0)
    y_bin = np.asarray(y)[:,None] == LABELS
    return {'accuracy':float(accuracy_score(y,pred)), 'precision_weighted':float(precision),
            'recall_weighted':float(recall), 'f1_weighted':float(f1),
            'macro_f1':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
            'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'roc_auc_macro_ovr':float(roc_auc_score(y,p,labels=LABELS,multi_class='ovr',average='macro')),
            'average_precision_macro_ovr':float(average_precision_score(y_bin,p,average='macro')),
            'brier_multiclass':float(np.mean(np.sum((p-y_bin)**2,axis=1))),
            'log_loss':float(log_loss(y,p,labels=LABELS))}

def digest(values):
    return hashlib.sha256(pd.util.hash_pandas_object(pd.Series(values),index=False).values.tobytes()).hexdigest()

def fit_cached(name, X, y, cache, signature, dummy=False):
    path = cache/(name+'.joblib')
    identity = dict(signature, name=name, rows=len(y), row_hash=digest(X.index), label_hash=digest(y.to_numpy()))
    if path.exists():
        saved = joblib.load(path)
        if saved['identity'] == identity:
            print('Reutilizado:',name,flush=True)
            return saved
    model = make_dummy() if dummy else make_pipeline()
    print('Ajustando:',name,'| casos:',len(y),flush=True)
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
        warnings.simplefilter('always')
        model.fit(X,y)
    nit = [] if dummy else model.named_steps['classifier'].n_iter_.tolist()
    converged = dummy or (max(nit)<PARAMS['max_iter'] and not any(issubclass(w.category,ConvergenceWarning) for w in caught))
    saved = {'identity':identity,'model':model,'iterations':nit,'converged':converged,
             'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in caught],
             'seconds':time.perf_counter()-start}
    joblib.dump(saved,path,compress=3)
    print('Completado:',name,'| convergió:',converged,'| iteraciones:',nit,flush=True)
    if not converged: raise RuntimeError('Debe corregirse la convergencia numérica antes de evaluar: '+name)
    return saved

def predict(model,X):
    with threadpool_limits(limits=2):
        p = model.predict_proba(X)
    assert list(model.classes_) == LABELS and np.isfinite(p).all()
    return p

def cm_metrics(cm):
    tp = np.diag(cm); support = cm.sum(axis=1); predicted = cm.sum(axis=0)
    recall = np.divide(tp,support,out=np.zeros(4),where=support!=0)
    f1 = np.divide(2*tp,support+predicted,out=np.zeros(4),where=(support+predicted)!=0)
    return {'accuracy':float(tp.sum()/cm.sum()),'balanced_accuracy':float(recall.mean()),'macro_f1':float(f1.mean())}

def bootstrap_groups(y, groups, probabilities, n_cm=1000, n_auc=200):
    """Bootstrap pareado por establecimiento, condicionado a modelos ya ajustados."""
    y = np.asarray(y)
    codes, unique = pd.factorize(groups,sort=True)
    G = len(unique)
    matrices = {}
    for name,p in probabilities.items():
        cell = (y-1)*4+p.argmax(axis=1)
        matrices[name] = np.bincount(codes*16+cell,minlength=G*16).reshape(G,4,4)
    rng = np.random.default_rng(SEED+100)
    records=[]; paired=[]; omitted=0
    for b in range(n_cm):
        counts = np.bincount(rng.integers(0,G,size=G),minlength=G)
        by_model = {}
        for name,mat in matrices.items():
            cm = np.einsum('g,gij->ij',counts,mat)
            if np.any(cm.sum(axis=1)==0): break
            by_model[name] = cm_metrics(cm)
        if len(by_model)!=len(matrices): omitted+=1; continue
        for name,values in by_model.items():
            for metric,value in values.items(): records.append((b,name,metric,value))
        for metric in ['accuracy','balanced_accuracy','macro_f1']:
            paired.append((b,metric,by_model['LogisticRegression'][metric]-by_model['DummyClassifier'][metric]))
        if b<n_auc:
            weights = counts[codes]
            values = {}
            for name,p in probabilities.items():
                value = float(roc_auc_score(y,p,labels=LABELS,multi_class='ovr',average='macro',sample_weight=weights))
                records.append((b,name,'roc_auc_macro_ovr',value)); values[name]=value
            paired.append((b,'roc_auc_macro_ovr',values['LogisticRegression']-values['DummyClassifier']))
        if (b+1)%200==0: print('Bootstrap por establecimiento:',b+1,'/',n_cm,flush=True)
    table = pd.DataFrame(records,columns=['replicate','model','metric','value'])
    out = table.groupby(['model','metric']).value.agg(lower=lambda x:x.quantile(.025),upper=lambda x:x.quantile(.975),replicates='count').reset_index()
    differences = pd.DataFrame(paired,columns=['replicate','metric','value']).groupby('metric').value.agg(lower=lambda x:x.quantile(.025),upper=lambda x:x.quantile(.975),replicates='count').reset_index()
    return {'intervals':out.to_dict('records'),'paired_differences':differences.to_dict('records'),
            'omitted_replicates':omitted,'seed':SEED+100,'establishments':G,
            'note':'IC percentiles del 95%; se remuestrean establecimientos completos, sin reajustar el modelo.'}

def run_study(data_path, screening_path, cache_dir):
    data_path, cache = Path(data_path),Path(cache_dir)
    cache.mkdir(exist_ok=True,parents=True)
    previous = json.loads(Path(screening_path).read_text(encoding='utf-8'))['baseline']
    signature = {'data_size':data_path.stat().st_size,'data_mtime_ns':data_path.stat().st_mtime_ns,
                 'holdout_groups_hash':digest(previous['test_establishment_ids']),'features':RAW_FEATURES,
                 'parameters':PARAMS,'sklearn':sklearn.__version__,'cv_folds':3,'fractions':[.25,.5,1.0],
                 'bootstrap_cm':1000,'bootstrap_auc':200,'seed':SEED,'design_version':1}
    report_path = cache/'evaluation.json'
    if report_path.exists():
        result = json.loads(report_path.read_text(encoding='utf-8'))
        if result['signature']==signature:
            print('Evaluación completa recuperada de la caché verificada; no se repiten ajustes.',flush=True)
            return result
    columns = RAW_FEATURES+['id','establishment_id','incident_outcome','date_of_incident']
    d = pd.read_csv(data_path,usecols=columns,dtype=str,encoding='latin1').replace(r'^\s*$',np.nan,regex=True)
    dates = pd.to_datetime(d.date_of_incident,format='%m/%d/%Y',errors='coerce')
    d = d.loc[dates.dt.year.eq(2023)].copy()
    held = d.establishment_id.isin(previous['test_establishment_ids'])
    tr,te = d.loc[~held],d.loc[held]
    X,y,g = tr[RAW_FEATURES],tr.incident_outcome.astype(int),tr.establishment_id
    X_test,y_test,g_test = te[RAW_FEATURES],te.incident_outcome.astype(int),te.establishment_id
    assert len(X)==previous['train_rows'] and len(X_test)==previous['test_rows']
    assert g.nunique()==previous['train_establishments'] and g_test.nunique()==previous['test_establishments']
    assert not set(g)&set(g_test)
    assert list(prepare_features(X.head()).columns)==previous['features']
    assert y.value_counts().sort_index().to_dict()=={int(k):v for k,v in previous['train_distribution'].items()}
    assert y_test.value_counts().sort_index().to_dict()=={int(k):v for k,v in previous['test_distribution'].items()}
    # El modelo final se ajusta sin consultar la reserva. Su evaluación se hace al final.
    final = fit_cached('final_lr',X,y,cache,signature)
    dummy_final = fit_cached('final_dummy',X,y,cache,signature,dummy=True)
    cv_path = cache/'grouped_validation.json'
    if cv_path.exists() and json.loads(cv_path.read_text())['signature']==signature:
        cross = json.loads(cv_path.read_text())
        print('Validación y curva de aprendizaje recuperadas.',flush=True)
    else:
        cv = StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=SEED)
        cv_rows,learning=[],[]
        for fold,(a,b) in enumerate(cv.split(X,y,g),1):
            fit_groups = g.iloc[a].unique()
            assert not set(fit_groups)&set(g.iloc[b])
            assert set(y.iloc[a])==set(y.iloc[b])==set(LABELS)
            order = np.random.default_rng(SEED+fold).permutation(np.sort(fit_groups))
            for fraction in [.25,.5,1.0]:
                selected = order[:max(1,int(np.ceil(len(order)*fraction)))]
                indices = a[g.iloc[a].isin(selected).to_numpy()]
                assert set(y.iloc[indices])==set(LABELS)
                saved = fit_cached(f'cv{fold}_fraction{int(100*fraction)}',X.iloc[indices],y.iloc[indices],cache,signature)
                train_scores = metrics(y.iloc[indices],predict(saved['model'],X.iloc[indices]))
                val_scores = metrics(y.iloc[b],predict(saved['model'],X.iloc[b]))
                base = {'fold':fold,'fraction':fraction,'train_rows':len(indices),'validation_rows':len(b),
                        'train_groups':len(selected),'validation_groups':int(g.iloc[b].nunique()),
                        'group_overlap':0,'iterations':saved['iterations'],'converged':saved['converged'],
                        'train_deaths':int(y.iloc[indices].eq(1).sum()),'validation_deaths':int(y.iloc[b].eq(1).sum())}
                learning.append(dict(base,train_metrics=train_scores,validation_metrics=val_scores))
                if fraction==1:
                    cv_rows.append(dict(base,model='LogisticRegression',metrics=val_scores))
                    dummy = fit_cached(f'cv{fold}_dummy',X.iloc[a],y.iloc[a],cache,signature,dummy=True)
                    cv_rows.append(dict(base,model='DummyClassifier',metrics=metrics(y.iloc[b],predict(dummy['model'],X.iloc[b]))))
            del saved
        cross = {'signature':signature,'cv':cv_rows,'learning_curve':learning}
        cv_path.write_text(json.dumps(cross,indent=2),encoding='utf-8')
    # Única evaluación final de las especificaciones fijadas de antemano.
    p_final,p_dummy = predict(final['model'],X_test),predict(dummy_final['model'],X_test)
    scores={}; reports={}; cms={}
    for name,p in [('LogisticRegression',p_final),('DummyClassifier',p_dummy)]:
        scores[name]=metrics(y_test,p)
        pred=np.asarray(LABELS)[p.argmax(axis=1)]
        reports[name]=classification_report(y_test,pred,labels=LABELS,output_dict=True,zero_division=0)
        cms[name]=confusion_matrix(y_test,pred,labels=LABELS).tolist()
    np.savez_compressed(cache/'holdout_predictions.npz',y=y_test.to_numpy(),groups=g_test.to_numpy(dtype=str),
                        case_ids=te.id.to_numpy(dtype=str),lr=p_final,dummy=p_dummy)
    boot = bootstrap_groups(y_test,g_test,{'LogisticRegression':p_final,'DummyClassifier':p_dummy})
    model = final['model']
    names = model.named_steps['onehot'].get_feature_names_out(FEATURES)
    coefficient = pd.DataFrame(model.named_steps['classifier'].coef_.T,index=names,columns=LABELS)
    coefficient.to_csv(cache/'coefficients.csv',encoding='utf-8')
    result={'signature':signature,'holdout_metrics':scores,'classification_reports':reports,'confusion_matrices':cms,
            'final_fit':{k:v for k,v in final.items() if k not in ['model','identity']},
            'cv':cross['cv'],'learning_curve':cross['learning_curve'],'bootstrap':boot,
            'train_rows':len(X),'test_rows':len(X_test),'train_groups':int(g.nunique()),'test_groups':int(g_test.nunique()),
            'test_death_groups':int(g_test[y_test.eq(1)].nunique()),'encoded_features':len(names),
            'raw_features':RAW_FEATURES,'features':FEATURES,'train_case_hash':digest(tr.id),'test_case_hash':digest(te.id)}
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Evaluación y ajustes completados.',flush=True)
    return result

if __name__=='__main__':
    # Guardar referencias al módulo importable, no al espacio temporal __main__.
    from osha_deliverable_utils import run_study as run_study_module
    root=Path(__file__).resolve().parent
    result=run_study_module(root/'ITA Case Detail Data 2023 through 12-31-2023OIICS.csv',
                     root.parent/'osha_screening_results.json',root/'osha_deliverable_results')
    print(json.dumps({'final_fit':result['final_fit'],'metrics':result['holdout_metrics']},ensure_ascii=False,indent=2))
