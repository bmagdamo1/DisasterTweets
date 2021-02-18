from scrape_event import ScrapeEvent

nashville_bombing = ScrapeEvent(
    ['2020-12-24', '2020-12-26'],
    ['nashville', 'los angeles', 'miami', 'chicago', 'philadelphia'],
    [1,            0,             0,       0,        0             ],
    scrape_job_iterations=20
)

beirut_explosion = ScrapeEvent(
    ['2020-08-03', '2020-08-05'],
    ['beirut', 'london', 'tel aviv', 'ottowa', 'johannesburg'],
    [1,         0,        0,         0,        0             ],
    scrape_job_iterations=20
)

brunswick_co_tornados = ScrapeEvent(
    ['2021-02-15', '2021-02-17'],
    ['"34.06923,-78.147"', 'atlanta', 'savannah', 'huntsville', 'raleigh'],
    [1,                     0,        0,          0,             0       ],
    scrape_job_iterations=20
)

nashville_bombing.scrape()
print(nashville_bombing.toPandas())
nashville_bombing.toPandas().to_csv("nashville_bombing_event.csv")

beirut_explosion.scrape()
print(beirut_explosion.toPandas())
beirut_explosion.toPandas().to_csv("beirut_explosion_event.csv")

brunswick_co_tornados.scrape()
print(brunswick_co_tornados.toPandas())
brunswick_co_tornados.toPandas().to_csv("brunswick_co_tornados_event.csv")
