import glob, os, numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, classification_report
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
labels = dict(l.strip().split(',') for l in open(R+'/eval_labels.csv').readlines()[1:])
cand = sorted(glob.glob(R+'/candidate_tiles/*/*.png')); ev = sorted(glob.glob(R+'/eval_set/*.png'))
def feat(p):
    a = np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)/255
    f=[a.mean((0,1)), a.std((0,1)), np.percentile(a.reshape(-1,3),[5,25,50,75,95],axis=0).ravel()]
    for c in range(3): f.append(np.histogram(a[...,c],bins=16,range=(0,1))[0]/4096)
    g = a.mean(2); gx=np.abs(np.diff(g,axis=1)); gy=np.abs(np.diff(g,axis=0))
    f.append([gx.mean(),gy.mean(),gx.std(),gy.std(),(gx>0.1).mean(),(gy>0.1).mean()])
    return np.concatenate([np.ravel(x) for x in f])
Xc=np.stack([feat(p) for p in cand]); yc=np.array([p.split('/')[-2] for p in cand])
Xe=np.stack([feat(p) for p in ev]); ye=np.array([labels[os.path.basename(p)] for p in ev])
for name,m in [('logreg',make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,C=1.0))),('rf',RandomForestClassifier(500,random_state=0))]:
    m.fit(Xc,yc); pr=m.predict(Xe); P=m.predict_proba(Xe); conf=P.max(1)
    print(f'== {name} eval acc {np.mean(pr==ye):.3f}')
    cls=m.classes_ if hasattr(m,'classes_') else m[-1].classes_
    print('classes', list(cls)); print(confusion_matrix(ye,pr,labels=cls))
    correct = pr==ye
    for t in [0.5,0.7,0.9]:
        k=conf>=t; print(f'  conf>={t}: coverage {k.mean():.2f}  acc-on-accepted {correct[k].mean() if k.any() else float("nan"):.3f}')
    print('  mean conf when correct %.2f / wrong %.2f'%(conf[correct].mean(),conf[~correct].mean()))
print(classification_report(ye,pr))
