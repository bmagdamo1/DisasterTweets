# Disaster Tweet Scraper
Based on Selenium, since twint seems not to be working

## Scraping API
We scrape tweets on an "event" basis. This means that, given a real-world disaster, we can scrape tweets from around that time in the city itself and several other "control" cities.

``` python
nashville_bombing = ScrapeEvent(
    ['2020-12-24', '2020-12-26'],
    ['nashville', 'los angeles', 'miami', 'chicago', 'philadelphia'],
    [1,            0,             0,       0,        0             ],
    scrape_job_iterations=20
)
```

## Output Format
To get the output as a pandas dataframe, call `nashville_bombing.toPandas()`

The table will look like

key | date | contents | city | city_in_disaster |
----|------|----------|------|------------------|
0 | 2020-12-25T15:41:14.000Z | Tweet (presumably) about the event | nashville | 1
1 | 2020-12-25T21:11:54.000Z | Probably not a disaster tweet | los angeles | 0