from pathlib import Path
ROOT=Path(r"C:\Progetto_AI")
SRC=ROOT/"05_rag_v5_9_anti.py"
DST=ROOT/"05_rag_v5_9_anti_v2.py"
text=SRC.read_text(encoding="utf-8-sig")
fn='''def query_specificity_adjustment(x, p):
    if p.get("intent") not in {"code_lookup", "code_pattern"} or "PA0001" not in p.get("tables", []):
        return 0.0, ""
    requested=set(str(v).upper() for v in (list(p.get("requested_fields", []))+list(p.get("explicit_fields", []))+list(p.get("functional_terms", []))))
    if "KOSTL" in requested:
        return 0.0, ""
    if any(z in p.get("concepts", []) for z in ("dati organizzativi","dati organizzativi dipendente")):
        return 0.0, ""
    typ=x.get("relationship_type","NONE")
    if typ=="SELECT_PA0001_WHERE_PERNR":
        return 35.0, "GENERIC_PA0001_PERNR_BONUS"
    if typ=="SELECT_KOSTL_FROM_PA0001_WHERE_PERNR":
        return -35.0, "UNREQUESTED_KOSTL_SPECIALIZATION_PENALTY"
    return 0.0, ""

'''
text=text.replace("\ndef rank(items, p):","\n"+fn+"def rank(items, p):",1)
old='''        else:
            score = se*25 + ex*2 + st*.3 + fs*.7 + fu*.5 + rel*1.0 - pen*.5

        uq = p["query"].upper().strip()
'''
new='''        else:
            score = se*25 + ex*2 + st*.3 + fs*.7 + fu*.5 + rel*1.0 - pen*.5

        specificity_delta, specificity_reason = query_specificity_adjustment(x, p)
        score += specificity_delta
        x["specificity_adjustment"] = specificity_delta
        x["specificity_reason"] = specificity_reason

        uq = p["query"].upper().strip()
'''
text=text.replace(old,new,1)
text=text.replace("print(\"SAP ABAP RAG TEST - V5.9 ANTI\")","print(\"SAP ABAP RAG TEST - V5.9 ANTI V2\")",1)
DST.write_text(text,encoding="utf-8")
print(DST)
