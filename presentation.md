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

However, even though education has grown rapidly in India, there hasn’t been a corresponding development in terms of labor market absorption, which varies from place to place (Government of India, Ministry of Education 2023a; Government of India, Ministry of Statistics and Programme Implementation 2023b). This discrepancy is visible among states and districts alike. In the present study, the combination of AISHE, PLFS, state economic variables, and proxy job postings data yields a basis for analysis. Two mismatch metrics are constructed and analyzed over time, after which district-level regressions are used to determine the reasons behind the regional variation. The result seems to be that skill profile, informality levels, and sectoral composition are key explanatory factors, while the simple linear approach proves the most effective for current data. While the approach can be useful in practice, some limitations should be noted. Namely, the available timeframe is limited, while part of the district metadata in CPERV1 is lacking.


Keywords: Education-employment mismatch; labor market alignment; India; district analytics; skill rate; informality

1. Introduction
It is evident that educational qualifications among Indian youth have become higher than ever before. This is rather obvious because more schooling means better employment opportunities. However, the other half of the issue – labor market – seems not be growing at the pace we expect. It appears that there is a certain mismatch where employers claim they cannot find suitable candidates fast enough while graduates continue changing low-level jobs regularly.

And this is what the current chapter is all about – identifying the mismatch between education preparation on the one hand and job demand on the other. As stated above, the former grows continuously, but the latter might increase, but could also stagnate, depending on a variety of factors. And while some discrepancies between states are visible, within each of them there are drastic differences between districts. Therefore, averaging national data would not show the real situation.

The current paper is developed as a diagnostic framework and not as a predictor. The aim is to measure the gap accurately and reveal reasons behind its inconsistency. This will involve bringing diverse public databases together and creating a uniform foundation to compute two mismatch indices as well as interpretative ML algorithms.

These contributions are applied, not theoretical:
1. An actionable approach to combining data on education, workforce participation, and economics.
2. Audit-proof mismatch measures subject to adjustment by policymakers.
3. District-based explanatory models with understandable features.
4. Output layer geared for presentation via dashboard.

This chapter is structured as follows. In section 2, we survey the literature and policy context. Section 3 details our methodology, including data, equations, algorithmic procedures, and modeling design. Section 4 examines empirical findings. Section 5 delves into policy implications and constraints. Section 6 concludes with directions for future research.


Figure 1. Concept Mind Map for Education-Employment Mismatch

                        


2. Literature Review / Related Work
Most academic scholarship on education-employment mismatch can be categorized into one of two general approaches. First, there is person-level mismatch that is associated with the phenomenon of either overeducation or underemployment (Quintini 2011; ILO 2022). Second, there is structural mismatch, in which the economy as a whole is not calibrated to accommodate changes in labor profiles smoothly (World Bank 2019; Mehrotra and Parida 2019). In India, both types exist, but the former type often prevails regionally.

There are three common structural frictions that Indian literature on the subject matter emphasizes time and again. First, skill formation and skill utilization do not always overlap seamlessly, particularly in case of smaller urban and rural-to-urban areas (Kumar and Chandra 2020). Second, labor informality causes signaling and productivity mismatch issues to arise, thus making credentialism less advantageous for many workers in India (NITI Aayog 2021). Third, sectoral composition is important, because more concentrated low-productivity sectors mean absorption but not necessarily matching (Mehrotra and Sinha 2017).

The simple fact is that there exists a straightforward data problem too. National surveys give a decent idea of large-scale trends, but policymaking requires more localized statistics. This study makes use of AISHE for educational statistics, PLFS for labor-related data, and economic state files to complement. What makes things difficult is that all three sources come in different formats. Names vary, years of data collection are different, and keys may not line up perfectly. The result is that many prior papers limit themselves to description alone. Nothing wrong with that, but poor integration will generally produce a poor modeling effort.

Prior studies on the use of machine learning techniques in labor economics have been geared primarily toward maximizing prediction accuracy. In such applications, however, the need for interpretability cannot be ignored. In cases when the outputs of the model must be used as input for further discussion involving district administrations and policy makers, a slightly simplified yet defensible solution may work better.

This paper represents the practical application of mismatch literature. Instead of theorizing, we attempt to combine the raw data in an effort to develop relevant indicators and assess possible explanations for observed variations.

3. Methodology / Proposed System

3.1 Data Sources and Variables
We utilized four data sources:

1. All India Survey on Higher Education (AISHE) to get enrollment and participation indicators as an education-side signal (Government of India, Ministry of Education 2023a).
2. Periodical Labour Force Survey (PLFS) to obtain the employment and structure indicators. We apply CPERV1 featureset, which refers to the cleaned version of district-level feature set used in this study (Government of India, Ministry of Statistics and Programme Implementation 2023b).
3. Economic indicators for the State – poverty, per capita, and youth unemployment context variable.
4. High-level job-posting proxy data (demand-side pressure).
Main predictor variables obtained from PLFS-CPERV1 are:
1. skill_rate – proportion of the population working with the relevant skill profile.
2. informal_rate – proportion of workers engaged in informal employment.
3. sector_share_agri, sector_share_industry, sector_share_services.

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
In general, it appears that the education index is increasing more rapidly than the employment index over the periods we currently observe. There are some spikes in the graph, possibly exaggerated ones. The reason for this may lie partly in the use of the min-max method on a limited number of observations. Nevertheless, the trend continues to suggest the increasing pressure for mismatch.

Divergence within states is significant. Some states with strong manufacturing or formal services sectors exhibit higher correlations between the education and employment indexes. There are other states with large positive gaps; this implies that the education sector values are relatively high compared to labor absorption measures.

4.2 Model Performance
District model metrics are shown in Table 1.

Table 1. District-Level Regression Performance

Model
R²
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
Baseline Model (Mean Predictor)
NA
3.9162


In terms of out-of-sample R2 and stability from train to test, Linear Regression is performing better, followed by RF and GBR. The latter two fit training data very well, but their results on the test set are not as pronounced, indicating possible overfitting. These numbers are based on the output in `data/eda/model_metrics_district.txt` file obtained for an April 2026 run.

Thus, in this case, the more complex model is less reliable. This might not be the case later when we obtain longer time periods and more features, but now we should rely on the simpler model.

4.3 Feature Interpretation
The following three features consistently stand out in terms of importance/coefficient values:
1. Positive correlation between informal_rate and the mismatch pressure;
2. Skill_rate has a positive effect on decreasing mismatch if the sector can absorb additional workers; for districts with poor structural conditions, the increase in skill alone is not enough;
3. Low-productivity sectors are negatively correlated with alignment.

But these are associations rather than cause-and-effect relationships. But the directionality is consistent with other research on the relationship between work and the economy and with experience in management. Training enrollments increase; placement depends on the local economic base.

4.4 Discussion of Uncertainty
A few caveats matter for interpretation.

1. Time-series length is limited for some components, so trend confidence is moderate, not strong.
2. District-name mapping is incomplete in one CPERV1 extract; code-level identifiers were used where names were missing.
3. Job-posting proxies can be urban-biased and may under-represent informal hiring channels.

Even with these limits, the framework is still useful as a ranking-and-diagnosis tool. It should not be read as a long-horizon forecasting engine at this stage.

5. Applications and Challenges
5.1 Practical Implementations
Chapter outputs could be utilized for multiple policy workflows.

1. Skill training prioritization: Areas with high gap_ratio and informal_rate could be identified as candidates for skill and placement partnerships rather than just skills.
2. State labor dashboard: State departments could monitor whether increases in education levels are accompanied by an increase in the quality of jobs year after year.
3. Sectoral planning: When sector shares suggest risks of concentration, governments could use targeted incentives, apprenticeships, and industry engagement at the local level.
4. Implementation evaluation: Managers of intervention programs could observe whether skill rates and absorption results improve simultaneously through interventions.

A straightforward visualization tool could increase implementation significantly. High-level policymakers will never start with the details of the model. What they need is three things right away: where the pressure is high, why, and how to act in the next quarter.

5.2 Challenges for Operation and Analysis
There are certain technical challenges and other non-technical ones.

1. Data normalization costs: Open datasets have their value; however, they differ in terms of keys and frequencies.
2. Proxy problems: Employment quality is not completely captured by statistics.
3. Lack of causality: The current approach captures variance; however, it fails to create the causation process.
4. Practicality of outcomes: Complicated results mean slow adoption.

There is another element related to political economy that can't be addressed through data alone. Places experiencing larger mismatches also face deficiencies in infrastructure and poor demand from firms, which exist beyond the jurisdiction of education departments. Therefore, any change requires cooperation between departments, which leads to delays.

6. Conclusion and Future Work
Educational mismatch in employment in India is measureable using public sources with fair accuracy and clear spatial variations. An effective integration, good visualization indicators and district-level models will provide enough information for early-stage decision making. Within present data limitations, Linear Regression remains the most effective and reliable algorithm for testing results. This does not mean that advanced algorithms cannot help but rather implies that they are less trustworthy at present.

Some future directions are quite clear, including:
1. Increasing the time span to avoid normalization and improve trend conclusions;
2. Improving quality of district-level data and geospatial analysis;
3. Introducing additional socioeconomic and institutional factors;
4. Testing more advanced techniques like panel or quasi-experiment designs.

In any case, one general conclusion is quite clear. Education growth and labor market expansion have to be monitored and correlated at once. If they diverge, intervention has to be rapid and local. The other key lesson from this exercise is that this framework operates as a diagnostic tool and not as a causal engine, and policy implications have to keep in mind.




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





