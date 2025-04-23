#!/usr/bin/env python3
import os, sys, glob, json
from datetime import datetime, timedelta, timezone

# Directories for input data (from environment variables or defaults)
TWITTER_DIR = os.getenv('TWITTER_DIR', './teacher_data/twitter_exports')
STAYFREE_DIR = os.getenv('STAYFREE_DIR', './teacher_data/stayfree_exports')

# Try to initialize a sentiment analysis pipeline (Japanese BERT model)
sentiment_pipeline = None
try:
    from transformers import pipeline
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="daigo/bert-base-japanese-sentiment"
    )
except Exception as e:
    print(f"[WARNING] Could not load transformers model: {e}")
    sentiment_pipeline = None

# Define a simple fallback sentiment classifier (keyword-based) if pipeline is not available
positive_keywords = ["嬉しい", "楽しい", "最高", "大好き", "最高", "感謝", "happy", "joy", "love"]
negative_keywords = ["悲しい", "辛い", "苦しい", "嫌い", "最悪", "怒り", "死にたい", "疲れた", "angry", "sad"]

def classify_sentiment(text: str) -> str:
    """Classify sentiment of the given text as 'positive', 'negative', or 'neutral'."""
    # Use the BERT sentiment pipeline if available
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text[:512])  # truncate to 512 tokens if very long
            if result:
                # Get the label and normalize to lowercase
                label = result[0]['label'].lower()
                if label in ["positive", "negative", "neutral"]:
                    return label
                # Some models might output labels in Japanese or different format; handle if needed
                if label in ["ポジティブ", "ポジティブだね"]:  # just an example if Japanese output
                    return "positive"
                if label in ["ネガティブ", "ネガティブだね"]:
                    return "negative"
                # If label not recognized, fall through to keyword method
        except Exception as e:
            print(f"[WARNING] Sentiment pipeline failed on text: {e}")
            # fallback to keyword method if pipeline fails for this text

    # Fallback keyword-based sentiment detection:
    text_lower = text.lower()
    pos = any(word in text for word in positive_keywords)
    neg = any(word in text for word in negative_keywords)
    if pos and not neg:
        return "positive"
    if neg and not pos:
        return "negative"
    return "neutral"

# Helper to convert epoch ms to local datetime
JST = timezone(timedelta(hours=9))  # Japan Standard Time
def to_jst_datetime(ms: int) -> datetime:
    """Convert a timestamp in milliseconds to a timezone-aware datetime in JST."""
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(JST)

# 1. Parse StayFree usage backup to get sleep duration per day
sleep_hours_by_date = {}  # store estimated sleep hours for each date (string)
base_target_by_date = {}  # store base target (60,70,80) for each date

# Find usage backup file(s) in STAYFREE_DIR
for filepath in glob.glob(os.path.join(STAYFREE_DIR, "*.usage_backup")):
    try:
        # Read the file as binary and extract JSON content (skipping any non-JSON header bytes)
        with open(filepath, "rb") as bf:
            data = bf.read()
            # Find the first curly brace to locate JSON start
            start_idx = data.find(b'{')
            json_str = data[start_idx:].decode('utf-8')
            usage_data = json.loads(json_str)
    except Exception as e:
        print(f"[ERROR] Could not parse StayFree backup file {filepath}: {e}")
        continue

    # The usage_data is expected to contain a 'stores' dict with 'sessions_2' list
    sessions = usage_data.get("stores", {}).get("sessions_2", [])
    # Split sessions by day boundaries and record usage intervals per day
    usage_intervals_by_date = {}  # dict of date -> list of (start, end) datetimes in that date
    for sess in sessions:
        start_dt = to_jst_datetime(sess.get("startedAt"))
        end_dt = to_jst_datetime(sess.get("endedAt"))
        # Ensure start_dt <= end_dt
        if end_dt < start_dt:
            end_dt = start_dt
        # If session spans multiple days, split at midnight of start_dt's day
        start_date = start_dt.date()
        end_date = end_dt.date()
        if start_date == end_date:
            usage_intervals_by_date.setdefault(str(start_date), []).append((start_dt, end_dt))
        else:
            # Session goes into the next day
            # End of start_date at 23:59:59...
            end_of_start_day = datetime(start_date.year, start_date.month, start_date.day, 23, 59, 59, tzinfo=JST)
            usage_intervals_by_date.setdefault(str(start_date), []).append((start_dt, end_of_start_day))
            # Beginning of next day at 00:00:00 to end_dt
            usage_intervals_by_date.setdefault(str(end_date), []).append((datetime(end_date.year, end_date.month, end_date.day, 0, 0, 0, tzinfo=JST), end_dt))
            # Note: Assuming sessions don't span more than 2 days continuously.

    # Now compute longest gap (in hours) for each day
    for date_str, intervals in usage_intervals_by_date.items():
        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        day_start = datetime.fromisoformat(date_str).replace(tzinfo=JST)
        day_end = (day_start + timedelta(days=1))
        longest_gap = 0.0
        prev_end = day_start  # start of day
        for (s_dt, e_dt) in intervals:
            # gap from prev_end to current start
            gap = (s_dt - prev_end).total_seconds() / 3600.0
            if gap > longest_gap:
                longest_gap = gap
            # move prev_end forward if this session ends later
            if e_dt > prev_end:
                prev_end = e_dt
        # gap from last usage end to end of day
        final_gap = (day_end - prev_end).total_seconds() / 3600.0
        if final_gap > longest_gap:
            longest_gap = final_gap

        sleep_hours_by_date[date_str] = longest_gap
        # Determine base target from sleep_hours
        if longest_gap >= 7.0:
            base_target_by_date[date_str] = 80
        elif longest_gap < 6.0:
            base_target_by_date[date_str] = 60
        else:
            base_target_by_date[date_str] = 70

# 2. Parse Twitter tweets.js data to get tweets per day and sentiment
tweets_by_date = {}   # dict of date_str -> list of (tweet_text, sentiment_label)
# Find all tweets.js files in TWITTER_DIR (in case of multiple parts)
for filepath in glob.glob(os.path.join(TWITTER_DIR, "**", "tweets*.js"), recursive=True):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read Twitter data file {filepath}: {e}")
        continue
    # Remove the JavaScript assignment part if present (e.g., "window.YTD.tweets.part0 = ")
    json_start = raw.find('[')
    json_end = raw.rfind(']')
    if json_start == -1 or json_end == -1:
        print(f"[WARNING] No JSON array found in {filepath}")
        continue
    tweets_json_str = raw[json_start:json_end+1]
    try:
        tweets = json.loads(tweets_json_str)
    except Exception as e:
        print(f"[ERROR] JSON parse failed for {filepath}: {e}")
        continue

    # Iterate through tweets
    for entry in tweets:
        tweet = entry.get("tweet", {})
        text = tweet.get("full_text", "")
        created_at = tweet.get("created_at")
        if not created_at or text is None:
            continue
        # Parse created_at (e.g. "Tue Mar 18 08:07:10 +0000 2025") and convert to JST date
        try:
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
        except Exception:
            # If format unexpected, skip
            continue
        dt_jst = dt.astimezone(JST)
        date_str = dt_jst.strftime("%Y-%m-%d")
        # Classify sentiment of the tweet content
        sentiment_label = classify_sentiment(text)
        # Store result
        tweets_by_date.setdefault(date_str, []).append((text, sentiment_label))

# 3. Determine sentiment adjustment per day and prepare CSV rows
rows = []
for date_str, base_target in base_target_by_date.items():
    # Determine overall day sentiment adjustment
    adjust = 0
    tweet_list = tweets_by_date.get(date_str, [])
    if tweet_list:
        # Count if any positive and any negative tweets on that day
        any_positive = any(lbl == "positive" for (_, lbl) in tweet_list)
        any_negative = any(lbl == "negative" for (_, lbl) in tweet_list)
        if any_positive and not any_negative:
            adjust = 5
        elif any_negative and not any_positive:
            adjust = -5
        else:
            adjust = 0
    else:
        # No tweets that day -> no sentiment adjustment
        adjust = 0
    final_target = base_target + adjust

    # Add a row for the StayFree (sleep) data of that day
    sleep_hours = sleep_hours_by_date.get(date_str, None)
    if sleep_hours is not None:
        # Format sleep hours to one decimal place
        content = f"睡眠時間（推定）: {sleep_hours:.1f}時間"
        rows.append((date_str, content, "stayfree", "neutral", final_target))
    # Add rows for each tweet on that day
    for (text, sent) in tweets_by_date.get(date_str, []):
        rows.append((date_str, text, "twitter", sent, final_target))

# 4. Write to CSV file
output_path = "/myhealth/data/health_features.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
try:
    with open(output_path, "w", encoding="utf-8") as csvfile:
        # Write header
        csvfile.write("date,content,source,sentiment,target\n")
        for (date_str, content, source, sentiment, target) in rows:
            # Escape quotes in content
            content_escaped = content.replace('"', '""')
            # Enclose content in quotes if it contains comma
            if ',' in content_escaped or '\n' in content_escaped:
                content_escaped = f'"{content_escaped}"'
            csvfile.write(f"{date_str},{content_escaped},{source},{sentiment},{target}\n")
    print(f"✅ Successfully generated CSV at {output_path}")
except Exception as e:
    print(f"[ERROR] Failed to write CSV: {e}")
