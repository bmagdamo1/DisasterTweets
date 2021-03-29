from scrape_event import ScrapeEvent

cities = ['christchurch','nashville','beirut','atlanta','fukishama','austin','moore','boston','nyc','joplin','houston','omaha']
dates = [['2019-03-14','2019-03-16'],['2020-12-24', '2020-12-26'],['2020-08-03', '2020-08-05'],['2021-03-15', '2021-03-17'],['2011-03-10', '2011-03-12'],
         ['2021-02-15', '2021-02-17'],['2013-05-19', '2013-05-21'],['2013-04-14', '2013-04-16'],['2012-10-30', '2012-11-01'],['2011-05-21', '2011-05-23'],
         ['2017-08-29', '2017-08-31'], ['2019-03-16', '2019-03-18']]
disaster_flags = [0]*len(cities)
num_disasters = len(cities)


for item in range(num_disasters):
    name = cities[item] + '_disaster'
    date = dates[item]
    disaster_flags[item]=1
    event = ScrapeEvent(date_range=date,cities=cities,disaster_flags=disaster_flags)
    event.scrape()
    event.toPandas().to_csv(name + '.csv')
    disaster_flags[item]=0 #Reset city flag
    print("Finished " + cities[item])




