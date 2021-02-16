from scrape_job import ScrapeJob

nashville_job = ScrapeJob('2020-12-24', '2020-12-26', 'nashville')
beirut_job = ScrapeJob('2020-08-03', '2020-08-05', 'beirut')

nashville_df = nashville_job.get_tweets(100)
beirut_df = beirut_job.get_tweets(100)

print("***** Nashville, TN *****")
print(nashville_df)
nashville_df.to_csv('nashville.csv')

print("***** Beirut, Lebanon *****")
print(beirut_df)
beirut_df.to_csv('beirut.csv')
