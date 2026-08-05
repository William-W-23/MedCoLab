#!/usr/bin/env python3
"""Fail-closed preflight for the fixed five-client FedBN experiments."""
import argparse,collections,csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fail(message): print(f"PREFLIGHT FAIL: {message}",file=sys.stderr); raise SystemExit(2)
def existing_parent(path):
    path=Path(path)
    while not path.exists(): path=path.parent
    return path
def audit_manifest(path, expected_sources, label_field, group_audit_unavailable):
    path=Path(path); manifest=json.loads(path.read_text()); rows=list(csv.DictReader(Path(manifest["source_manifest"]).open()))
    group_split_leaks=manifest.get("group_split_leaks")
    if group_split_leaks is None:
        if not group_audit_unavailable: fail("group_split_leaks is unavailable")
        group_audit={"status":"unavailable","reason":"source patient/slide identifiers are not published"}
    else:
        if group_audit_unavailable: fail("group audit marked unavailable but manifest provides group_split_leaks")
        if int(group_split_leaks)!=0: fail("group_split_leaks != 0")
        group_audit={"status":"available","group_split_leaks":0}
    table=[]; hist=[]
    for client in range(5):
        block=manifest["clients"][str(client)]; sets=[set(block[f"{s}_paths"]) for s in ("train","val","test")]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)): fail(f"client{client} split overlap")
        subset=[r for r in rows if r["client"]==str(client)]; sources=collections.Counter(r["dataset"] for r in subset)
        if len(sources)!=expected_sources or any(v<=0 for v in sources.values()): fail(f"client{client} missing source: {sources}")
        classes=collections.Counter(r[label_field] for r in subset if r["split"]!="ssl"); hist.append(dict(classes))
        table.append({"client":client,"sources":dict(sources),"source_splits":dict(collections.Counter(r["split"] for r in subset)),"actual_train":len(sets[0]),"actual_val":len(sets[1]),"actual_test":len(sets[2])})
    if all(x==hist[0] for x in hist[1:]): fail("client class distributions are exactly balanced")
    return {"manifest":str(path),"manifest_sha256":sha(path),"group_audit":group_audit,"clients":table,"class_histograms":hist}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--task",required=True); p.add_argument("--experiment-id",required=True); p.add_argument("--manifest",required=True); p.add_argument("--expected-manifest-sha",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--bn-dir",required=True); p.add_argument("--workdir",required=True); p.add_argument("--rounds",type=int,required=True); p.add_argument("--local-epochs",type=int,required=True); p.add_argument("--num-clients",type=int,required=True); p.add_argument("--participation",type=float,required=True); p.add_argument("--client-lr",type=float,required=True); p.add_argument("--server-lr",type=float,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--expected-sources",type=int,default=5); p.add_argument("--label-field",default="label"); p.add_argument("--required-free-gib",type=float,default=8.0); p.add_argument("--group-audit-unavailable",action="store_true"); p.add_argument("--smoke",action="store_true"); a=p.parse_args()
    if a.task!='classification' or a.num_clients!=5 or a.participation!=1.0 or a.local_epochs!=1: fail("fixed protocol mismatch")
    if str(Path(a.bn_dir)).startswith('/tmp'): fail("BN directory is under /tmp")
    for x in (a.output_dir,a.bn_dir,a.workdir):
        if Path(x).exists(): fail(f"path already exists: {x}")
    if sha(a.manifest)!=a.expected_manifest_sha: fail("manifest SHA256 mismatch")
    free=shutil.disk_usage(existing_parent(a.output_dir)).free; required=(1.0 if a.smoke else a.required_free_gib)*1024**3
    if free<required: fail(f"insufficient free space: {free/1024**3:.2f} GiB < {required/1024**3} GiB")
    root=Path(__file__).resolve().parents[1]; dirty=subprocess.check_output(['git','-C',str(root),'status','--short'],text=True); commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    print(json.dumps({**vars(a),"use_fedbn":True,"bn_policy":"local_fedbn","server_optimizer":"fedyogi","git_commit":commit,"git_dirty":bool(dirty.strip()),"git_status":dirty.splitlines(),"free_gib":free/1024**3,"data_audit":audit_manifest(a.manifest,a.expected_sources,a.label_field,a.group_audit_unavailable)},indent=2,ensure_ascii=False))
if __name__=='__main__': main()
