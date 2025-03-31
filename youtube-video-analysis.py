# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset from GitHub
github_url = "https://raw.githubusercontent.com/Emstan32/youtube-video-analysis/main/youtube_data_extended.csv"
df = pd.read_csv(github_url)

# Convert 'upload_date' to datetime format
df['upload_date'] = pd.to_datetime(df['upload_date'])

# Handle missing values (fill with median for numeric, mode for categorical)
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# Feature Engineering
df["engagement_rate"] = (df["likes"] + df["comments"]) / df["views"]  # Engagement Rate
df["popularity_index"] = (df["views"] * 0.5) + (df["likes"] * 0.3) + (df["comments"] * 0.2)

# Extract Upload Day and Month
df["upload_day"] = df["upload_date"].dt.day_name()
df["upload_month"] = df["upload_date"].dt.month_name()

# Display dataset info
print(df.info())
print(df.head())

#  Top 10 Most Viewed Videos
top_videos = df.nlargest(10, "views")
plt.figure(figsize=(10,5))
sns.barplot(x=top_videos["views"], y=top_videos["title"], palette="coolwarm")
plt.title("Top 10 Most Viewed Videos")
plt.xlabel("Views")
plt.ylabel("Video Title")
plt.show()

#  Engagement Rate Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["engagement_rate"], bins=30, kde=True, color="purple")
plt.title("Engagement Rate Distribution")
plt.xlabel("Engagement Rate")
plt.ylabel("Frequency")
plt.show()

#  Watch Time vs. Views Relationship
plt.figure(figsize=(10,5))
sns.scatterplot(x=df["watch_time"], y=df["views"], hue=df["trend_video"], palette=["gray", "red"])
plt.title("Watch Time vs Views")
plt.xlabel("Watch Time (minutes)")
plt.ylabel("Views")
plt.legend(["Non-Trending", "Trending"])
plt.show()

#  Trending Videos by Category
trending_videos = df[df["trend_video"] == True]
plt.figure(figsize=(10,5))
sns.boxplot(x="category", y="views", data=trending_videos, palette="Oranges")
plt.title("Views Distribution of Trending Videos by Category")
plt.xticks(rotation=30)
plt.show()

# Best Upload Day Analysis (Which day has the most views?)
daywise_views = df.groupby("upload_day")["views"].sum().sort_values(ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(x=daywise_views.index, y=daywise_views.values, palette="Blues")
plt.title("Total Views by Upload Day")
plt.xlabel("Day of the Week")
plt.ylabel("Total Views")
plt.xticks(rotation=30)
plt.show()

#  Trending Video Trends Over Time
plt.figure(figsize=(12,5))
sns.countplot(x="upload_month", data=trending_videos, order=df["upload_month"].value_counts().index, palette="viridis")
plt.title("Number of Trending Videos by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()

#  Trending vs. Non-Trending Video Views Comparison
plt.figure(figsize=(10,5))
sns.boxplot(x="trend_video", y="views", data=df, palette="Set2")
plt.xticks([0,1], ["Non-Trending", "Trending"])
plt.title("Trending vs. Non-Trending Video Views")
plt.show()

# Trending vs. Non-Trending Likes & Comments
plt.figure(figsize=(12,5))
sns.boxplot(x="trend_video", y="likes", data=df, palette="Blues")
plt.xticks([0,1], ["Non-Trending", "Trending"])
plt.title("Trending vs. Non-Trending Video Likes")
plt.show()

plt.figure(figsize=(12,5))
sns.boxplot(x="trend_video", y="comments", data=df, palette="Oranges")
plt.xticks([0,1], ["Non-Trending", "Trending"])
plt.title("Trending vs. Non-Trending Video Comments")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df[["views", "likes", "comments", "watch_time", "engagement_rate", "popularity_index"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
