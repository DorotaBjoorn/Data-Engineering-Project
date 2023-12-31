import json
import os
import re
from pathlib import Path

import dash
import pandas as pd
from dash import Input, Output, State
from dash.dependencies import Input, Output

from newsfeed.dashboard_article_item import (
    dashboard_content_container,
    news_artcle_div,
    title_heading_for_dashboard,
)
from newsfeed.dashboard_layout import layout
from newsfeed.utils import (
    NEWS_ARTICLES_ARTICLE_SOURCES,
    NEWS_ARTICLES_SUMMARY_SOURCES,
    SWEDISH_NEWS_ARTICLES_SUMMARY_SOURCES,
    formated_source,
    source_dict,
)

app = dash.Dash(
    __name__,
    meta_tags=[dict(name="viewport", content="width=device-width, initial-scale=1.0")],
)
app.layout = layout

server = app.server


# json -> dict -> [dict, dict, dict] -> df where each dict is a row
def read_json_files_to_df(folder_path):
    df_list = []
    # os.listdir lists all files in directory specified by folder_path
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            with open(os.path.join(folder_path, filename), "r") as f:
                data = json.load(f)
                df_list.append(data)
    return pd.DataFrame(df_list)


# This function returns a blog articles from either a specific source or from all
# depending on the arguments inputed in the function when the function is called
def get_news_data(news_blog_source="all_blogs", language="english", max_num_articles=15):
    # Check which language user press to determine which data to use
    if language == "english":
        source_data = NEWS_ARTICLES_SUMMARY_SOURCES
    elif language == "swedish":
        source_data = SWEDISH_NEWS_ARTICLES_SUMMARY_SOURCES
    else:
        raise ValueError("Oops! Only english and swedish are the only supported languages!")

    if news_blog_source not in source_dict:
        raise ValueError("Invalid choice. Use 'mit', 'google_ai', 'ai_blog', 'open_ai' or 'all_blogs'")

    if news_blog_source == "all_blogs":
        df_list = []
        for source in source_dict["all_blogs"]:
            # limits the total articles to 15 before concatinating into one dataframe
            temp_df = read_json_files_to_df(source_data[source])[:max_num_articles]
            temp_df["source"] = formated_source[source]
            df_list.append(temp_df)
        return pd.concat(df_list, ignore_index=True)

    # If choice is just a single blog, then only 15 id returned
    df = read_json_files_to_df(source_data[source_dict[news_blog_source]])[:max_num_articles]
    df["source"] = formated_source[source_dict[news_blog_source]]
    return df


def fetch_and_prepare_articles(language, df):
    articles_sources = NEWS_ARTICLES_ARTICLE_SOURCES

    all_articles_df = pd.concat(
        [read_json_files_to_df(articles_sources[key]) for key in articles_sources]
    )

    required_cols = ["unique_id", "link", "published"]
    if not all(col in all_articles_df.columns for col in required_cols):
        raise ValueError(
            "Missing required columns unique_id, link and published date in articles dataframe"
        )

    news_item_with_date = []
    for index, row in df.iterrows():
        title = row["title"]
        summary_technical = row["blog_summary_technical"]
        summary_non_technical = row["blog_summary_non_technical"]
        article_source = row["source"]
        unique_id = row["unique_id"]

        # Look up the additional data based on the unique_id
        additional_data = all_articles_df[all_articles_df["unique_id"] == unique_id]

        # Fetch link and published date if the unique_id is found
        if not additional_data.empty:
            link = additional_data.iloc[0]["link"]
            published_date = additional_data.iloc[0]["published"]

            news_item_with_date.append(
                news_artcle_div(
                    title=title,
                    published_date=published_date,
                    technical_summary=summary_technical,
                    non_technical_summary=summary_non_technical,
                    link=link,
                    language=language,
                    article_source=article_source,
                )
            )
        else:
            raise ValueError(f"No matching additional info for Id: {unique_id}")
    return news_item_with_date


@app.callback(Output("blogs-df", "data"), [Input("data-type-dropdown", "value")])
def blogs_df(selected_data_type):
    # Get the news data based on the selected type
    news_data = get_news_data(selected_data_type)
    # Convert the DataFrame to a dictionary of records and return
    return news_data.to_dict("records")


@app.callback(
    Output("language-store", "data"),
    [Input("btn-english", "n_clicks"), Input("btn-swedish", "n_clicks")],
    [State("language-store", "data")],
)
def update_language(n_clicks_english, n_clicks_swedish, data):
    ctx = dash.callback_context
    # check if button is clicked and which button recieved a click event
    if not ctx.triggered_id or ctx.triggered_id == "None":
        raise dash.exceptions.PreventUpdate
    if "btn-english" in ctx.triggered_id:
        data["language"] = "english"
    elif "btn-swedish" in ctx.triggered_id:
        data["language"] = "swedish"
    return data


@app.callback(
    [Output("blog-heading", "children"), Output("content-container", "children")],
    [
        Input("dropdown-choice", "value"),
        Input("language-store", "data"),
        Input("search-btn", "n_clicks"),
    ],
    [State("blogs-df", "data"), State("search-input", "value")],
)
def display_blogs(choice, language_data, n_clicks, blogs_data, search_query):
    language = language_data.get("language", "english")
    if search_query:
        max_articles_to_return = 100
    else:
        max_articles_to_return = 15
    news_data = get_news_data(choice, language=language, max_num_articles=max_articles_to_return)

    if (
        "title" not in news_data.columns
        or "blog_summary_technical" not in news_data.columns
        or "unique_id" not in news_data.columns
    ):
        return "No title", "No Summary"

    news_item_with_date = fetch_and_prepare_articles(language, news_data)
    sorted_news_item_with_date = sorted(news_item_with_date, key=lambda x: x["date"], reverse=True)

    if search_query:
        pattern = re.compile(search_query.replace(" ", "[-_ ]?"), re.IGNORECASE)
        sorted_news_item_with_date = [
            item
            for item in sorted_news_item_with_date
            if pattern.search(str(item["div"].children[0].children))
        ]

    # Extract the sorted divs
    sorted_news_item = [item["div"] for item in sorted_news_item_with_date]

    heading = title_heading_for_dashboard(heading="The Midjourney Journal")
    content = dashboard_content_container(sorted_news_item)
    return heading, content


if __name__ == "__main__":
    app.run_server(debug=True)
