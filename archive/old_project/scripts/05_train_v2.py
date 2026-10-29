"""Retrain CNN + BiLSTM with label smoothing, weight decay, better early stopping."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import yaml
from Bio import SeqIO
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

CFG = yaml.safe_load(open("config.yaml"))
SEED = CFG["project"]["seed"]
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
FAMILIES = CFG["models"]["families"]; FAM2IDX = {f:i for i,f in enumerate(FAMILIES)}; N = len(FAMILIES)
CKPT = Path(CFG["training"]["checkpoint_dir"]); CKPT.mkdir(exist_ok=True)
RES  = Path(CFG["training"]["results_dir"]);  RES.mkdir(exist_ok=True)
LS = CFG["training"]["label_smoothing"]
WD = CFG["training"]["weight_decay"]
DEV = "cpu"
print(f"device: {DEV}  label_smoothing: {LS}  weight_decay: {WD}")


def load_split():
    df = pd.read_csv("data/processed/master_final.csv")
    cache, out = {}, {"train":[], "val":[], "test":[]}
    for _, r in df.iterrows():
        p = r["sequence_file_path"]
        if p not in cache: cache[p] = {rec.id.split(".")[0]: str(rec.seq) for rec in SeqIO.parse(p,"fasta")}
        s = cache[p].get(r["accession"].split(".")[0])
        if s: out[r["split"]].append((r["accession"], s.upper(), r["family"]))
    for k,v in out.items(): print(f"  {k}: {len(v)} seqs, {len({x[2] for x in v})} families")
    return out


def windows(seq, wlen, stride):
    if len(seq) < wlen: yield seq + "N"*(wlen-len(seq)); return
    for i in range(0, len(seq)-wlen+1, stride): yield seq[i:i+wlen]
    if (len(seq)-wlen) % stride: yield seq[-wlen:]


DNA = {"A":0,"C":1,"G":2,"T":3}
def onehot(s):
    x = np.zeros((4,len(s)),dtype=np.float32)
    for i,c in enumerate(s):
        if c in DNA: x[DNA[c],i]=1.0
    return x


def kmers(s,k):
    ids=[]
    for i in range(0,len(s)-k+1,k):
        idx,ok=0,True
        for c in s[i:i+k]:
            if c not in "ACGT": ok=False; break
            idx=idx*4+"ACGT".index(c)
        ids.append(idx if ok else 0)
    return ids or [0]


class DS(Dataset):
    def __init__(self, recs, enc, wlen, stride):
        self.items=[]
        for si,(_,s,fam) in enumerate(recs):
            ws=list(windows(s,wlen,stride)); w=1.0/max(len(ws),1); y=FAM2IDX[fam]
            for win in ws: self.items.append((enc(win),y,w,si))
        print(f"    {len(recs)} seqs -> {len(self.items)} windows")
    def __len__(self): return len(self.items)
    def __getitem__(self,i): return self.items[i]


def col_onehot(b):
    x,y,w,s=zip(*b); return (torch.from_numpy(np.stack(x)),torch.tensor(y),torch.tensor(w),torch.tensor(s))
def col_kmers(b):
    x,y,w,s=zip(*b); L=max(len(t) for t in x); pad=np.zeros((len(x),L),dtype=np.int64)
    for i,t in enumerate(x): pad[i,:len(t)]=t
    return (torch.from_numpy(pad),torch.tensor(y),torch.tensor(w),torch.tensor(s))


class CNN(nn.Module):
    def __init__(self,f1,f2,k,dr):
        super().__init__()
        self.c1=nn.Conv1d(4,f1,k,padding="same"); self.b1=nn.BatchNorm1d(f1)
        self.c2=nn.Conv1d(f1,f2,k,padding="same"); self.b2=nn.BatchNorm1d(f2)
        self.pool=nn.AdaptiveMaxPool1d(1); self.drop=nn.Dropout(dr); self.fc=nn.Linear(f2,N)
    def forward(self,x):
        x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x)))
        return self.fc(self.drop(self.pool(x).squeeze(-1)))


class BiLSTM(nn.Module):
    def __init__(self,vocab,e,h,dr):
        super().__init__()
        self.emb=nn.Embedding(vocab,e,padding_idx=0)
        self.lstm=nn.LSTM(e,h,batch_first=True,bidirectional=True)
        self.drop=nn.Dropout(dr); self.fc=nn.Linear(h*2,N)
    def forward(self,x):
        out,_=self.lstm(self.emb(x)); return self.fc(self.drop(out.mean(1)))


@torch.no_grad()
def evaluate(model, loader, recs):
    model.eval(); probs=[[] for _ in recs]
    for x,y,w,s in loader:
        p=F.softmax(model(x),-1).numpy()
        for i,si in enumerate(s.tolist()): probs[si].append(p[i])
    yt,yp=[],[]
    for si,(_,_,fam) in enumerate(recs):
        if not probs[si]: continue
        yt.append(FAM2IDX[fam]); yp.append(int(np.argmax(np.mean(np.stack(probs[si]),0))))
    macro=f1_score(yt,yp,average="macro",zero_division=0)
    p,r,f1,_=precision_recall_fscore_support(yt,yp,labels=range(N),zero_division=0)
    return {"macro_f1":float(macro),"per_class_f1":[float(v) for v in f1],
            "labels_present": sorted({FAMILIES[i] for i in yt}),
            "n":len(yt)}


def train(name, model, tr_rec, va_rec, te_rec, ds_fn, collate, lr, bs, epochs, patience):
    print(f"\n=== {name} ===")
    tr_loader=DataLoader(ds_fn(tr_rec),batch_size=bs,shuffle=True,collate_fn=collate)
    va_loader=DataLoader(ds_fn(va_rec),batch_size=bs,shuffle=False,collate_fn=collate)
    te_loader=DataLoader(ds_fn(te_rec),batch_size=bs,shuffle=False,collate_fn=collate)
    model=model.to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=WD)
    best_f1,best_state,wait=-1.0,None,0
    for ep in range(1,epochs+1):
        model.train(); tot,n=0.0,0
        for x,y,w,s in tr_loader:
            x,y,w=x.to(DEV),y.to(DEV),w.to(DEV)
            ce=F.cross_entropy(model(x),y,reduction="none",label_smoothing=LS)
            loss=(ce*w).sum()/w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
            tot+=loss.item()*len(torch.unique(s)); n+=len(torch.unique(s))
        m=evaluate(model,va_loader,va_rec)
        print(f"  ep {ep:2d}  loss {tot/n:.4f}  val_macro_f1 {m['macro_f1']:.4f}")
        if m["macro_f1"]>best_f1+1e-4:
            best_f1=m["macro_f1"]; wait=0
            best_state={k:v.clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=patience: print(f"  early stop at epoch {ep}"); break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(),CKPT/f"{name}.pt")
    val_m=evaluate(model,va_loader,va_rec)
    test_m=evaluate(model,te_loader,te_rec)
    print(f"  saved {CKPT}/{name}.pt")
    print(f"  VAL  macro_f1={val_m['macro_f1']:.4f}  (5 families in-distribution)")
    print(f"  TEST macro_f1={test_m['macro_f1']:.4f}  (3 families held out)")
    return {"val":val_m,"test":test_m}


def main():
    print("loading...")
    sp=load_split()
    results={}
    c=CFG["models"]["cnn"]
    m=CNN(c["filters"][0],c["filters"][1],c["kernel"],c["dropout"])
    results["cnn"]=train("cnn",m,sp["train"],sp["val"],sp["test"],
        lambda r: DS(r,onehot,c["window"],c["stride"]),col_onehot,
        c["lr"],c["batch_size"],c["epochs"],c["patience"])
    b=CFG["models"]["bilstm"]
    m2=BiLSTM(b["vocab"],b["embed"],b["hidden"],b["dropout"])
    results["bilstm"]=train("bilstm",m2,sp["train"],sp["val"],sp["test"],
        lambda r: DS(r,lambda s: np.array(kmers(s,b["kmer"])),b["window"],b["stride"]),col_kmers,
        b["lr"],b["batch_size"],b["epochs"],b["patience"])
    results["_excluded"]={"model":"DNABERT-2",
        "reason":"Custom ALiBi incompatible with PyTorch 2.14 on CPU",
        "impact":"Cross-architecture comparison uses CNN and BiLSTM"}
    results["_config"]={"label_smoothing":LS,"weight_decay":WD,
        "test_species":["MERS","Ebola","Dengue"],
        "test_families":["Coronaviridae","Filoviridae","Flaviviridae"],
        "val":"20% stratified from training species"}
    json.dump(results,open(RES/"train_metrics.json","w"),indent=2)
    print(f"\nwrote {RES}/train_metrics.json")
    for k,v in results.items():
        if k.startswith("_"): continue
        print(f"  {k:8s}  VAL={v['val']['macro_f1']:.4f}  TEST={v['test']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
