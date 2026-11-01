"""Provider-level feature engineering for Medicare claims fraud detection."""
from pathlib import Path
import numpy as np
import pandas as pd

SUFFIX = {"train": "1542865627584", "unseen": "1542969243754"}
CHRONIC = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure", "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer", "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart", "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis", "ChronicCond_stroke",
]


def load_split(raw_dir, split):
    """Load labels, beneficiary, inpatient and outpatient tables for a split."""
    raw_dir, s = Path(raw_dir), SUFFIX[split]
    pre = "Train" if split == "train" else "Unseen"
    labels = pd.read_csv(raw_dir / f"{pre}-{s}.csv")
    bene = pd.read_csv(raw_dir / f"{pre}_Beneficiarydata-{s}.csv", parse_dates=["DOB", "DOD"])
    ip = pd.read_csv(raw_dir / f"{pre}_Inpatientdata-{s}.csv",
                     parse_dates=["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"])
    op = pd.read_csv(raw_dir / f"{pre}_Outpatientdata-{s}.csv",
                     parse_dates=["ClaimStartDt", "ClaimEndDt"])
    return labels, bene, ip, op


def _prep_bene(bene):
    b = bene.copy()
    ref = pd.Timestamp("2009-12-31")
    b["Age"] = (ref - b["DOB"]).dt.days / 365.25
    b["IsDeceased"] = b["DOD"].notna().astype(int)
    b["Renal"] = (b["RenalDiseaseIndicator"].astype(str) == "Y").astype(int)
    for c in CHRONIC:                      # source coding: 1 = yes, 2 = no
        b[c] = (b[c] == 1).astype(int)
    b["ChronicCount"] = b[CHRONIC].sum(axis=1)
    return b[["BeneID", "Age", "IsDeceased", "Renal", "ChronicCount", "State",
              "IPAnnualReimbursementAmt", "OPAnnualReimbursementAmt"]]


def _prep_claims(ip, op, bene):
    ip, op = ip.copy(), op.copy()
    ip["IsInpatient"], op["IsInpatient"] = 1, 0
    claims = pd.concat([ip, op], ignore_index=True)
    claims["StayDays"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days + 1
    dx = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
    px = [c for c in claims.columns if c.startswith("ClmProcedureCode_")]
    claims["NDiag"] = claims[dx].notna().sum(axis=1)
    claims["NProc"] = claims[px].notna().sum(axis=1)
    claims["NoAttending"] = claims["AttendingPhysician"].isna().astype(int)
    claims["HasOperating"] = claims["OperatingPhysician"].notna().astype(int)
    claims["HasOther"] = claims["OtherPhysician"].notna().astype(int)
    return claims.merge(_prep_bene(bene), on="BeneID", how="left"), dx


def build_provider_features(ip, op, bene):
    claims, dx = _prep_claims(ip, op, bene)
    g = claims.groupby("Provider")
    f = pd.DataFrame({
        "n_claims": g.size(),
        "ip_share": g["IsInpatient"].mean(),
        "reimb_sum": g["InscClaimAmtReimbursed"].sum(),
        "reimb_mean": g["InscClaimAmtReimbursed"].mean(),
        "reimb_max": g["InscClaimAmtReimbursed"].max(),
        "reimb_std": g["InscClaimAmtReimbursed"].std().fillna(0),
        "deductible_mean": g["DeductibleAmtPaid"].mean(),
        "stay_days_mean": g["StayDays"].mean(),
        "n_diag_mean": g["NDiag"].mean(),
        "n_proc_mean": g["NProc"].mean(),
        "no_attending_rate": g["NoAttending"].mean(),
        "operating_rate": g["HasOperating"].mean(),
        "other_phys_rate": g["HasOther"].mean(),
        "n_benes": g["BeneID"].nunique(),
        "n_attending": g["AttendingPhysician"].nunique(),
        "n_states": g["State"].nunique(),
        "bene_age_mean": g["Age"].mean(),
        "bene_deceased_rate": g["IsDeceased"].mean(),
        "bene_renal_rate": g["Renal"].mean(),
        "bene_chronic_mean": g["ChronicCount"].mean(),
        "bene_ip_annual_mean": g["IPAnnualReimbursementAmt"].mean(),
        "bene_op_annual_mean": g["OPAnnualReimbursementAmt"].mean(),
    })
    f["claims_per_bene"] = f["n_claims"] / f["n_benes"]
    f["claims_per_physician"] = f["n_claims"] / f["n_attending"].clip(lower=1)
    f["reimb_per_bene"] = f["reimb_sum"] / f["n_benes"]

    # Same beneficiary seen repeatedly by the same provider/physician pair
    pair = claims.groupby(["Provider", "BeneID", "AttendingPhysician"]).size()
    f["repeat_pair_rate"] = (pair > 1).groupby("Provider").mean()
    # Same-day repeat claims for one beneficiary at one provider (possible duplicate billing)
    dup = claims.groupby(["Provider", "BeneID", "ClaimStartDt"]).size()
    f["same_day_dup_rate"] = (dup > 1).groupby("Provider").mean()
    # Concentration: share of claims handled by the busiest physician
    top = claims.groupby(["Provider", "AttendingPhysician"]).size().groupby("Provider").max()
    f["top_physician_share"] = top / f["n_claims"]
    # Diagnosis diversity
    long_dx = claims[["Provider"] + dx].melt(id_vars="Provider").dropna(subset=["value"])
    f["dx_unique"] = long_dx.groupby("Provider")["value"].nunique()
    f["dx_unique_ratio"] = f["dx_unique"] / f["n_claims"]
    return f.fillna(0).reset_index()
