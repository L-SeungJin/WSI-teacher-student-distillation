import os, glob
import numpy as np
import pandas as pd

WSI_LIST="/workspace/my_exp/slides_27.txt"
RES_ROOT="/workspace/GC_WSI_Results_MoE_UNI"
OUT_ROOT="/workspace/my_exp/teacher"
os.makedirs(OUT_ROOT, exist_ok=True)

def main():
    ok=0
    for svs in open(WSI_LIST):
        sid=os.path.basename(svs.strip()).replace(".svs","")
        sdir=os.path.join(RES_ROOT, sid)
        cands=glob.glob(os.path.join(sdir, "*patch_preds*uni*.csv"))
        if not cands:
            print("[MISS]", sid, "no patch_preds csv in", sdir)
            continue
        csv_path=cands[0]
        df=pd.read_csv(csv_path)

        coords=df[["x","y"]].to_numpy(dtype=np.int32)
        pred=df["pred_idx"].to_numpy(dtype=np.int64)

        out_dir=os.path.join(OUT_ROOT, sid)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "coords.npy"), coords)
        np.save(os.path.join(out_dir, "teacher_pred.npy"), pred)
        ok += 1

    print("done. converted:", ok)

if __name__=="__main__":
    main()
