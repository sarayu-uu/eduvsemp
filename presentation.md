  INTERNSHIP REPORT-1

ON
Education-Employment Mismatch in India: Trends, Hotspots, and Structural Drivers


BY
Sarayu Ramdas
22STUCHH010195

AT
Bilvantis Technologies
An Internship Program II










Faculty of Science & Technology,
IFHE University
APRIL 2026












ABSTRACT

Education in India has expanded quickly, but labor-market absorption has not kept the same pace everywhere (Government of India, Ministry of Education 2023a; Government of India, Ministry of Statistics and Programme Implementation 2023b). That mismatch now shows up clearly across states and districts. In this report, I combine AISHE, PLFS, state economic indicators, and job-posting proxy data to build one working analysis base. Two mismatch measures are calculated and compared over time, then district-level models are trained to explain why the gap changes by location. The pattern is fairly consistent: skill mix, informality, and sector structure matter a lot, and the simpler linear model performs best on current data. The framework is useful for practical targeting, but there are limits too. Time coverage is short, and district metadata is still incomplete in part of the CPERV1 extract.


Keywords: Education-employment mismatch; labor market alignment; India; district analytics; skill rate; informality

1. Introduction
Education levels in India have clearly improved in recent years, and that part is easy to see. More students in schools and colleges should translate into stronger job outcomes. But the labor market side is not moving in a straight line. Many employers still say they cannot find the right fit fast enough, while many young graduates keep cycling through short-term or low-quality work.

That is the mismatch this chapter talks about. In simple terms, education readiness is rising, but job absorption is not always rising with it. Sometimes it does, sometimes it does not. The pattern shifts by state, and even within a state, one district can look very different from the next. So using only national averages can hide the actual stress points.

This work is built as a diagnostic setup, not a forecasting engine. The goal here is to measure the gap properly and explain why it differs across places. We merge scattered public datasets into one comparable base, calculate two mismatch indicators, and then test interpretable machine learning (ML) models to see what explains district-level variation better.

The contributions are practical rather than theoretical:
1. A reproducible pipeline for integrating education, labor, and economic data.
2. Transparent mismatch indicators that can be audited and modified by policymakers.
3. District-level explanatory modeling with clear feature meaning.
4. A dashboard-ready output layer for communication.

The chapter is organized as follows. Section 2 reviews related literature and policy context. Section 3 explains data, equations, algorithmic workflow, and model setup. Section 4 discusses empirical results. Section 5 covers policy applications and limitations. Section 6 closes with future work.

Figure 1. Concept Mind Map for Education-Employment Mismatch

                           [Education-Employment Mismatch]
                                      /     |      \
                                     /      |       \
                              [Skills]  [Informality]  [Sector Mix]
                                  \         |          /
                                   \        |         /
                                [Regional Variation]
                                         |
                                   [Policy Targeting]


2. Literature Review / Related Work
Most research on education-employment mismatch falls into two broad lines. One line studies person-level issues, like overeducation or underemployment (Quintini 2011; ILO 2022). The other line looks at structural frictions, where the economy itself is not set up to absorb changing labor profiles evenly (World Bank 2019; Mehrotra and Parida 2019). India shows both patterns, but across regions, structural mismatch often stands out more.

Indian labor-market literature repeatedly points to three linked frictions. One, skill formation and skill utilization do not line up cleanly, especially across smaller urban centers and rural transition zones (Kumar and Chandra 2020). Two, high informality distorts signaling and productivity matching, which weakens the return to credentials in many occupations (NITI Aayog 2021). Three, sector concentration matters: districts with larger low-productivity service and informal trade shares often absorb people into work, but not necessarily into stable or well-matched work (Mehrotra and Sinha 2017).

There is a plain data challenge as well. National reports are good for big-picture trends, but policy action usually needs district-level detail. AISHE covers education well, PLFS is strong on labor outcomes, and state economic files add context. The problem is they do not arrive in one clean format. Names differ, year coverage is uneven, and keys do not always match neatly. So a lot of studies stop at descriptive summaries. Fair enough, but weak integration usually means weak modeling later.

Prior studies using ML in labor economics tend to optimize prediction accuracy, sometimes with opaque feature interactions. For policy, pure accuracy is not enough. Interpretable models are often more useful, particularly when the output needs to support district-level discussions with administrators and program teams. A model that is slightly less complex but easier to defend can be the better tool.

This report sits in that applied space. It takes ideas from mismatch literature, but the focus here is practical: combine messy data, define usable indicators, test what explains variation, and be clear about where uncertainty still remains.

3. Methodology / Proposed System

3.1 Data Sources and Variables
Four grouped sources were used:

1. All India Survey on Higher Education (AISHE): enrollment and participation indicators for education-side signal (Government of India, Ministry of Education 2023a).
2. Periodic Labour Force Survey (PLFS): employment and structure indicators. We use the CPERV1 feature set; CPERV1 is the cleaned district-level feature bundle used in this project (Government of India, Ministry of Statistics and Programme Implementation 2023b).
3. State economic indicators: poverty, per-capita proxy, and youth unemployment context variables.
4. Job-posting proxy data: high-level demand pressure indicator.
Key district predictors from PLFS-CPERV1:
1. skill_rate: share of workers with relevant skill profile.
2. informal_rate: share of workers in informal employment.
3. sector_share_agri, sector_share_industry, sector_share_services: labor distribution across major sectors.

3.2 Data Integration Pipeline
Data integration followed six operational steps:

1. Standardize state names, district identifiers, and year formats.
2. Build a state-year base table from AISHE and economic indicators.
3. Join PLFS-CPERV1 district features using harmonized keys.
4. Compute normalized education and employment indices.
5. Derive mismatch measures (gap, gap_ratio).
6. Export analytical outputs for model training and dashboard usage.

A compact pseudocode form is provided below.

Algorithm 1. Unified Mismatch Data Builder

Input: AISHE table A, PLFS-CPERV1 table P, Economic table E, Job proxy J
Output: District-level analysis table D

1. Clean labels in A, P, E, J (state names, year, district code)
2. Build state_year_table S = merge(A, E, on = [state, year])
3. Build district_table T = merge(P, J, on = [state, district, year], left join)
4. Compute education_index and employment_index in S and T
5. Compute mismatch metrics gap and gap_ratio
6. Remove duplicates, flag missingness, and keep analysis-ready rows
7. Return D = T with selected predictors + target gap_ratio


Figure 2. Unified Data Pipeline and Modeling Flow

[AISHE]   [PLFS-CPERV1]   [Economic Indicators]   [Job Proxy]
    \            |                |                   /
     \           |                |                  /
      +-------- Data Cleaning and Key Standardization --------+
                               |
                        State-Year Merge
                               |
                       Feature Engineering
                               |
                   gap and gap_ratio Construction
                               |
                      Train/Test Model Stage
                               |
                LR / RF / GBR Comparison (R2, RMSE)
                               |
                      Dashboard + Policy Notes


3.3 Indicator Construction
Two core indicators are used.

Education and employment signals are normalized via min-max scaling:

(1) education_index_i = (E_i - E_min) / (E_max - E_min)

(2) employment_index_i = (L_i - L_min) / (L_max - L_min)

Absolute index gap:

(3) gap_i = education_index_i - employment_index_i

Ratio-based pressure indicator (district target variable):

(4) gap_ratio_i = education_index_i / max(epsilon, employment_rate_i)

where epsilon is a small constant to avoid division instability in very low employment-rate observations.

3.4 Model Design
We estimate district-level mismatch intensity using supervised regression:

(5) y_i = f(X_i) + eta_i

where y_i = gap_ratio_i, and X_i includes skill, informality, and sector-share predictors.

Models tested:
1. Linear Regression (LR)
2. Random Forest Regressor (RF)
3. Gradient Boosting Regressor (GBR)

Train-test split: 80/20 with fixed random seed for reproducibility.

Evaluation metrics:

(6) R2 = 1 - [sum_i (y_i - y_hat_i)^2] / [sum_i (y_i - y_bar)^2]

(7) RMSE = sqrt[(1/n) * sum_i (y_i - y_hat_i)^2]

A mean-predictor baseline is included to avoid overstating model value.

3.5 Architecture and Implementation Notes
Figure 1 and Figure 2 together summarize the architecture and flow from source integration to model output.

Technical implementation was done in Python with separate modules for cleaning, exploratory analysis, and app serving. The system is intentionally modular so that future data additions do not require full pipeline rewrites.

4. Results and Discussion

4.1 Descriptive Pattern
At the aggregate level, the education index seems to rise faster than the employment index in the years we currently have. Some jumps in the chart look sharp, maybe too sharp. Part of that comes from min-max scaling over a short series, so not every spike should be read as a real structural break. Even then, the broad direction still points to growing mismatch pressure.

State-level divergence is substantial. Some regions with stronger industrial or formal service bases show tighter education-employment alignment. Others show large positive gaps, meaning education-side values look relatively better than labor absorption indicators. District dispersion inside the same state is often wide, which suggests state-average targeting can miss local pressure pockets.

4.2 Model Performance
District model metrics are shown in Table 1.

Table 1. District-Level Regression Performance


Model


R2
RMSE
Linear Regression
0.7770
1.8491
Random Forest
0.7639
1.9026
Gradient Boosting
0.7620
1.9103
Baseline Model(mean predictor)
NA
3.9162


Linear Regression gives the best out-of-sample R2 and, just as important, it stays relatively stable from train to test. RF and GBR learn the training pattern strongly, but test gains are smaller, which hints at mild overfitting with the current feature set. These values come from the project run outputs in `data/eda/model_metrics_district.txt` (April 2026 run).

So for this dataset, the simpler model is doing the more reliable job. That may change later if we add longer time coverage and richer district covariates, but right now the linear model is the safer choice.

4.3 Feature Interpretation
Feature importance and coefficient patterns show three consistent drivers:
1. Higher informal_rate is generally associated with higher mismatch pressure.
2. Better skill_rate tends to reduce mismatch in districts where formal absorption exists; in structurally weak local markets, skill gains alone are not enough.
3. Sector composition matters. Higher low-productivity concentration often corresponds with weaker alignment.

These are associations, not strict causal claims. The direction, however, matches broader labor-economy literature and administrative experience. Program teams usually see this on the ground: training numbers go up, placement quality varies sharply by local economic structure.

4.4 Discussion of Uncertainty
A few caveats matter for interpretation.

1. Time-series length is limited for some components, so trend confidence is moderate, not strong.
2. District-name mapping is incomplete in one CPERV1 extract; code-level identifiers were used where names were missing.
3. Job-posting proxies can be urban-biased and may under-represent informal hiring channels.

Even with these limits, the framework is still useful as a ranking-and-diagnosis tool. It should not be read as a long-horizon forecasting engine at this stage.

5. Applications and Challenges

5.1 Practical Applications
The chapter outputs can support several policy workflows.

1. Targeted skilling allocation: Districts with high gap_ratio and high informal_rate can be prioritized for blended interventions (skills plus placement partnerships), not training-only programs.
2. State labor strategy dashboards: State departments can monitor whether education expansion is matched by employment quality improvements year to year.
3. Sector-specific planning: Where sector shares indicate concentration risks, governments can align incentives, apprenticeship programs, and local industry outreach.
4. Evaluation support: Program managers can track whether interventions move both skill rates and absorption outcomes together.

A simple visualization layer improves adoption a lot. Senior officials usually do not want model internals first. They want three things quickly: where pressure is high, what may be driving it, and what can be done next quarter.

5.2 Operational and Research Challenges
Some constraints are technical, and some are organizational.

1. Data harmonization overhead: Public datasets are valuable but inconsistent in keys and update cycles.
2. Proxy limitations: Employment quality is not fully captured by headline rates.
3. Causality gap: The current setup explains variation but does not establish causal pathways.
4. Administrative usability: If outputs are too complex, implementation slows down.

There is also a political-economy layer that data by itself cannot fix. Districts with higher mismatch often deal with infrastructure gaps and weak firm demand, and those sit outside education departments. So real action needs coordination across departments, and that is usually where delays begin.

6. Conclusion and Future Work
Education-employment mismatch in India can be measured reasonably well with available public data, and the pattern is clearly uneven across regions. A clean integration pipeline, readable indicators, and interpretable district models are enough to support early policy decisions. Under current data limits, Linear Regression stays the strongest and most stable model on test performance. That does not mean advanced models are useless. It just means the simpler one is more dependable for now.

Future extensions are already clear:
1. Expand time coverage to reduce normalization artifacts and strengthen trend inference.
2. Improve district metadata and geospatial joins for clearer local reporting.
3. Add richer socioeconomic covariates and institutional variables.
4. Test panel and quasi-experimental strategies for stronger causal interpretation.

The larger takeaway is direct: education expansion and labor absorption need to be tracked together, not in separate silos. When they drift apart, response has to be local, quick, and tied to actual district constraints. Also, this is still a diagnostic system, not a causal proof engine, so policy use should stay grounded and iterative.


References 

Government of India, Ministry of Education. 2023a. All India Survey on Higher Education (AISHE) 2021-22. New Delhi (IN): Department of Higher Education.
Government of India, Ministry of Statistics and Programme Implementation. 2023b. Periodic Labour Force Survey (PLFS): Annual Report 2022-23. New Delhi (IN): National Statistical Office.
International Labour Organization (ILO). 2022. Global Employment Trends for Youth 2022: Investing in Transforming Futures for Young People. Geneva (CH): ILO.
Kumar A, Chandra S. 2020. Skills, employability, and labor-market transitions in India: regional evidence. Indian Journal of Labour Economics. 63(4):941-960.
Mehrotra S, Parida J. 2019. Why is the unemployment rate so high in India? Economic and Political Weekly. 54(6):15-23.
Mehrotra S, Sinha S. 2017. Explaining falling female employment in India. World Development. 98:360-380.
NITI Aayog. 2021. Reforms in Urban Planning Capacity in India: Final Report. New Delhi (IN): Government of India.
Quintini G. 2011. Right for the Job: Over-qualified or Under-skilled? OECD Social, Employment and Migration Working Papers No. 120. Paris (FR): OECD Publishing.
World Bank. 2019. The Changing Nature of Work: World Development Report 2019. Washington (DC): World Bank.


