import torch
from transformers import AutoTokenizer, AutoModel

# Standard BERT, no trust_remote_code needed
name = "armheb/DNA_bert_6"
print(f"Loading {name}...")

try:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name)
    model.eval()
    
    seq = "ATCG" * 25
    enc = tok(seq, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc).last_hidden_state
        
    print(f"\n✅ SUCCESS! Standard DNA BERT loaded perfectly.")
    print(f"   Hidden dim: {out.shape[-1]}")
except Exception as e:
    print(f"\n❌ FAILED: {e}")