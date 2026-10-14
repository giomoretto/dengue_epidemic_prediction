this is Data Dictionary that describes the various entries of the dataset and their meaning
data_ini_SE : First day of the epidemiological week (Sunday)
EW: Epidemiological week
casos_est : Estimated number of cases per week using the nowcasting model (note: values ​​are retrospectively updated every week)
cases_est_min and cases_est_max: Intervalo de credibilidade de 95% do número estimado de casos
cases: Number of notified cases per week (values ​​are retrospectively updated every week)
p_rt1: Probability (Rt > 1). For issuing the orange alert, we use criterion p_rt1 > 0.95 for 3 weeks or more.
p_inc100k: Estimated incidence rate (cases per pop x 100.000)
Localidade_id: Sub-municipality division (currently implemented only in Rio de Janeiro)
level:: Alert level (1 = green, 2 = yellow, 3 = orange, 4 = red), more details, see (Know more)
id: Numeric index
versao_modelo: Model version (internal use)
Rt: Point estimate of the reproductive number of cases, see Learn more
pop: Estimated population (IBGE)
tempmin: Average of the daily minimum temperatures along the week
tempmean: Average daily temperatures along the week
tempmax: Average of the daily maximum temperatures along the week
humidmin: Average of the daily minimum relative air humidity along the week
humidmean: Average daily relative air humidity along the week
humidmax: Average of the daily maximum relative air humidity along the week
receptive: Indicates climate receptivity, i.e., conditions for high vectorial capacity. 0 = unfavorable, 1 = favorable, 2 = favorable this week and last week, 3 = favorable for at least three weeks (sufficient for completing a transmission cycle)
transmission: Evidence of sustained transmission: 0 = no evidence, 1 = possible, 2 = likely, 3 = highly likely
nivel_inc:: Estimated incidence below pre-epidemic threshold, 1 = above pre-epidemic threshold but below epidemic threshold, 2 = above epidemic threshold
