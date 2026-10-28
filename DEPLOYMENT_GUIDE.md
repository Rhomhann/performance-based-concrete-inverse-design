# Deployment Guide — Industry Evaluation Edition

## Recommended deployment
Streamlit Community Cloud + GitHub.

## Repository contents
Upload these files to the root of one GitHub repository:

- streamlit_app.py
- app_industry.py
- concrete_inverse_design_engine_v33.py
- Concrete_Data_V1.csv
- requirements.txt
- README_INDUSTRY_EDITION.md
- .gitignore

`streamlit_app.py` is the recommended deployment entrypoint.

## Streamlit Community Cloud
1. Sign in to Streamlit Community Cloud and connect the GitHub account containing the repository.
2. Choose **Create app** → **Yup, I have an app**.
3. Select the repository and branch (`main` is recommended).
4. Set the entrypoint to `streamlit_app.py`.
5. In Advanced settings, use a supported Python version. Python 3.12 is the current Community Cloud default and is recommended for this deployment.
6. Choose a memorable app subdomain if available.
7. Deploy.
8. Test at least:
   - 50 MPa, 28 days, SCM permitted
   - 50 MPa, 28 days, Cement only
   - Results tab
   - all Analysis panels
   - w/cm scenario analysis
   - Qualification tab
   - CSV export

## Before sending to evaluators
- Open the deployed URL in an incognito/private browser window.
- Confirm no local Windows paths appear.
- Confirm no source-code traceback is shown during normal use.
- Confirm the research-prototype and laboratory-qualification notices are visible.
- Record the deployed URL and deployment date.
- Do not change the frozen V3.3 engine during the evaluation round unless a genuine defect is discovered.
