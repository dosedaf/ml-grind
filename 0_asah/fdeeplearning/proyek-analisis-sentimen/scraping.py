from google_play_scraper import reviews, Sort, reviews_all
import pandas as pd

app_id = 'com.linkedin.android' 

result, continuation_token = reviews(
    app_id,
    lang='id',          
    country='id',
    sort=Sort.NEWEST,    
    count=30000,         
    filter_score_with=None
)

df = pd.DataFrame(result)

df = df[['content', 'score']]
df.rename(columns={'content': 'text', 'score': 'rating'}, inplace=True)

# Simpan hasil scrape
df.to_csv('linkedin_reviews.csv', index=False)

print(f"Scraped {len(df)} reviews")