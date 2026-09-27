import glob, os, random
from PIL import Image, ImageDraw
R = os.environ.get('DATASET_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'be-mlsys-assignment-dataset'))
S = os.path.dirname(os.path.abspath(__file__))
random.seed(1)
classes = sorted(os.listdir(R+'/candidate_tiles'))
n=12; s=96; pad=4; lw=110
W = lw + n*(s+pad); H = len(classes)*(s+pad)
M = Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(M)
for r,c in enumerate(classes):
    ps = random.sample(sorted(glob.glob(f'{R}/candidate_tiles/{c}/*.png')), n)
    d.text((4, r*(s+pad)+s//2), c, fill='black')
    for i,p in enumerate(ps):
        M.paste(Image.open(p).convert('RGB').resize((s,s),Image.NEAREST),(lw+i*(s+pad), r*(s+pad)))
M.save(S+'/montage_candidates.png')
# eval montage with labels in order
labels=[l.strip().split(',') for l in open(R+'/eval_labels.csv').readlines()[1:]]
M2 = Image.new('RGB',(lw+n*(s+pad), len(classes)*(s+pad)),'white'); d=ImageDraw.Draw(M2)
by={}
for f,c in labels: by.setdefault(c,[]).append(f)
for r,c in enumerate(classes):
    d.text((4, r*(s+pad)+s//2), c, fill='black')
    for i,f in enumerate(random.sample(by[c],n)):
        M2.paste(Image.open(f'{R}/eval_set/{f}').convert('RGB').resize((s,s),Image.NEAREST),(lw+i*(s+pad), r*(s+pad)))
M2.save(S+'/montage_eval.png')
# odd tiles
odd=['candidate_tiles/SeaLake/SeaLake_18.png','candidate_tiles/Industrial/Industrial_106.png','candidate_tiles/AnnualCrop/AnnualCrop_60.png','candidate_tiles/SeaLake/SeaLake_55.png']
M3=Image.new('RGB',(len(odd)*(192+pad),192),'white')
for i,p in enumerate(odd): M3.paste(Image.open(f'{R}/{p}').convert('RGB').resize((192,192),Image.NEAREST),(i*(192+pad),0))
M3.save(S+'/odd.png')
