import pandas as pd

from scrape_job import ScrapeJob

class ScrapeEvent:

    def __init__(
        self,
        date_range,
        cities,
        disaster_flags,
        scrape_job_iterations = 20
    ):
    
        self.__start = date_range[0]
        self.__end = date_range[1]
        self.__cities = cities
        self.__disaster_flags = disaster_flags
        self.__scrape_job_iterations = scrape_job_iterations
        
        self.__df = None
        
    def scrape(self):
        
        # In case we've run scrape multiple times,
        # clear the dataframe since we concat to it
        self.__df = None
        
        for i, city in enumerate(self.__cities):
            
            # Get the tweets
            job = ScrapeJob(self.__start, self.__end, city)
            tweets = job.get_tweets(self.__scrape_job_iterations)
            
            # Add label columns for the city name and disaster flag
            tweets['city'] = city
            tweets['city_in_disaster'] = self.__disaster_flags[i]
            
            # Add these into the shared dataframe
            self.__df = pd.concat([self.__df, tweets])
            
        
    def toPandas(self):
        return self.__df
