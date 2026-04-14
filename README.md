# Education–Employment Mismatch (India)

## Problem Statement
Analyze and quantify the mismatch between educational output and labor market demand across Indian states. Identify whether unemployment is driven by inadequate job opportunities, skill mismatches, or broader economic conditions.

## Dataset Links
Add the official sources here:
- AISHE enrollment tables: `TBD`
- Unemployment (CMIE/PLFS or equivalent): `TBD`
- Economic indicators (GDP, poverty, literacy, labor force): `TBD`
- Job market postings dataset: `TBD`

## How To Run
```bash
pip install -r requirements.txt
python cleaning/cleaning.py
python eda/eda.py
streamlit run app.py
```

## Key Result (Current)
Random Forest R2 ≈ 0.21.  
Labor participation and employment levels dominate feature importance; education alone is not a strong reducer of unemployment in this dataset.
