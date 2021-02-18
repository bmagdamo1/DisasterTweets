import json
import time
import hashlib
import traceback


import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.keys import Keys

class ScrapeJob:

    def __init__(self, start, end, near):

        self.start = start
        self.end = end
        self.near = near

    def get_tweets(self, iterations=3):

        option = webdriver.ChromeOptions()
        option.add_argument(" — incognito")
        browser = webdriver.Chrome(executable_path="./chromedriver", chrome_options=option)

        tweet_contents = []
        tweet_dates = []

        try:
        
            print("Opening: " + self.__get_twitter_query_url(
                self.start, self.end, self.near
            ))
        
            # Navigate to the URL and wait for
            # the page to load and the search
            # to execute with clientside js
            browser.get(self.__get_twitter_query_url(
                self.start, self.end, self.near
            ))
            time.sleep(2)
            
            # We need the body element to inject PAGE_DOWN keys
            body = browser.find_element_by_tag_name('body')

            # The loop parameter decides roughly how many tweets
            # we'll get
            for _ in range(iterations):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.3)
                
                container = browser.find_elements_by_xpath(self.__get_tweets_container_xpath())

                # Add the tweets to the arrays
                for c in container:
                
                    try:
                        
                        date = self.__get_tweet_date(c)
                        content = self.__get_tweet_contents(c)
                        
                        print(date)
                        print(content)
                        print()
                    
                        tweet_dates.append(self.__get_tweet_date(c))
                        tweet_contents.append(self.__get_tweet_contents(c))
                        
                    except: break
                    
        except:
            traceback.print_exc()

        finally:
        
        
            # Check for errors
            if len(tweet_dates) is not len(tweet_contents):
                print("tweet_dates: " + str(len(tweet_dates)))
                print("tweet_contents: " + str(len(tweet_contents)))
                raise Exception()
                    
            # Show our progress
            print(".", end="")
            
            # Rebuild the dataframe with what we currently have
            df = pd.DataFrame({
                'date': tweet_dates,
                'contents': tweet_contents
            }, columns=['date', 'contents'])
            df = df.drop_duplicates()
        
            browser.quit()
            return df
            
    def __make_tweet_dict_object(self, date, contents):
        obj = {
            "date": date,
            "contents": contents
        }
        key = hashlib.md5(json.dumps(obj).encode('utf-8')).hexdigest()
        
        return key, obj
            
    def __get_tweets_container_xpath(self):
        return '//article/div/div/div'
        # return '//article/div/div/div/div[2]/div[2]/div[2]/div[1]/div'

    def __get_tweet_contents(self, container):
        return container.find_element_by_xpath('./div[2]/div[2]/div[2]/div[1]/div').text
        
    def __get_tweet_date(self, container):
        return container.find_element_by_xpath('./div[2]/div[2]/div[1]/div/div/div[1]/a/time').get_attribute('datetime')

    def __get_twitter_query_url(self, since, until, near):
        return 'https://twitter.com/search?q=lang%3Aen%20until%3A' + until + '%20since%3A' + since + '%20-filter%3Alinks%20-filter%3Areplies%20near%3A' + near + '&src=typed_query'

