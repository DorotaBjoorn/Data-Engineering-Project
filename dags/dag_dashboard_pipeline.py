from datetime import datetime

from airflow.decorators import dag, task

from newsfeed import (
    download_blogs_from_rss,
    extract_articles,
    summarize,
    translate,
)

blogs_list = ["ai_blog", "mit", "open_ai"]


@task(task_id="download_blogs")
def download_blogs_from_rss_task() -> None:
    for blog in blogs_list:
        download_blogs_from_rss.main(blog_name=blog)


@task(task_id="extract_blogs")
def extract_blogs_task() -> None:
    for blog in blogs_list:
        extract_articles.main(blog_name=blog)


@task(task_id="summarize_blogs")
def summarize_blogs_task() -> None:
    for blog in blogs_list:
        summarize.main(blog_name=blog, model_type="local_model")


@task(task_id="translate_blogs")
def translate_blogs_task() -> None:
    for blog in blogs_list:
        translate.main(blog_name=blog)


@dag(
    dag_id="dashboard_pipline",
    start_date=datetime(2023, 6, 2),
    schedule_interval="@daily",
    catchup=False,
)
def pipeline():
    (
        download_blogs_from_rss_task()
        >> extract_blogs_task()
        >> summarize_blogs_task()
        >> translate_blogs_task()
    )


# Register the DAG
pipeline_instance = pipeline()
