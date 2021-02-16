import twint

from prompts import prompt_limit, prompt_output_file_name

def main():

    c = twint.Config()
    c.Search = "tornado"
    c.Store_csv = True
    c.Lang = "en"

    c.Retweets = True
    c.Replies = True

    c.Limit = prompt_limit()
    c.Output = prompt_output_file_name()

    # 10km around Nashville
    c.Geo = "36.16784,-86.77816,10km"

    c.Since = "2020-12-24"
    c.Until = "2020-12-26"

    twint.run.Search(c)

if __name__ == "__main__":
    main()