import dash_bootstrap_components as dbc
from dash import dcc, html

layout = dbc.Container(
    [
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    className="mb-44",
                    children=[
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            children=[
                                                html.Img(
                                                    id="midjourney-logo",
                                                    src="assets/midjourney-logo.png",
                                                    style={
                                                        "position": "absolute",
                                                        "top": "-2%",
                                                        "left": "-2%",
                                                        "width": "250px",
                                                    },
                                                ),
                                            ]
                                        )
                                    )
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    dbc.Row(
                                                        id="blog-heading",
                                                        style={"text-align": "center"},
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                ),
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardBody(
                                            dbc.Row(
                                                children=[
                                                    dbc.Col(
                                                        dbc.Card(
                                                            html.Div(
                                                                [
                                                                    dbc.CardGroup(
                                                                        children=[
                                                                            dcc.Dropdown(
                                                                                id="data-type-dropdown",
                                                                                className="hidden",
                                                                                options=[
                                                                                    {
                                                                                        "label": "All Blogs",
                                                                                        "value": "all_blogs",
                                                                                    },
                                                                                    {
                                                                                        "label": "Google-ai",
                                                                                        "value": "google_ai",
                                                                                    },
                                                                                    {
                                                                                        "label": "MIT",
                                                                                        "value": "mit",
                                                                                    },
                                                                                    {
                                                                                        "label": "AI-Blog",
                                                                                        "value": "ai_blog",
                                                                                    },
                                                                                ],
                                                                                value="all_blogs",
                                                                                style={
                                                                                    "width": "300px",
                                                                                    "height": "0px",
                                                                                },
                                                                            ),
                                                                        ],
                                                                    ),
                                                                ],
                                                                style={"justify-content": "center"},
                                                            ),
                                                            style={
                                                                "display": "none"
                                                            },  # This line hides the entire card
                                                        ),
                                                    ),
                                                    dbc.Col(
                                                        dbc.Card(
                                                            html.Div(
                                                                [
                                                                    dbc.CardGroup(
                                                                        children=[
                                                                            dbc.Card(
                                                                                dbc.CardBody(
                                                                                    children=[
                                                                                        dbc.Row(
                                                                                            children=[
                                                                                                dbc.Col(
                                                                                                    dbc.Label(
                                                                                                        "Select a Blog",
                                                                                                        html_for="dropdown-choice",
                                                                                                        style={
                                                                                                            "fontSize": "18px",
                                                                                                            "fontFamily": "Roboto",
                                                                                                            "marginBottom": "10px",
                                                                                                        },
                                                                                                    ),
                                                                                                    width={
                                                                                                        "size": 3
                                                                                                    },
                                                                                                ),
                                                                                                dbc.Col(
                                                                                                    dcc.Dropdown(
                                                                                                        id="dropdown-choice",
                                                                                                        options=[
                                                                                                            {
                                                                                                                "label": "All Blogs",
                                                                                                                "value": "all_blogs",
                                                                                                            },
                                                                                                            {
                                                                                                                "label": "Google-ai",
                                                                                                                "value": "google_ai",
                                                                                                            },
                                                                                                            {
                                                                                                                "label": "MIT",
                                                                                                                "value": "mit",
                                                                                                            },
                                                                                                            {
                                                                                                                "label": "AI-Blog",
                                                                                                                "value": "ai_blog",
                                                                                                            },
                                                                                                        ],
                                                                                                        value="all_blogs",
                                                                                                        style={
                                                                                                            "width": "300px"
                                                                                                        },
                                                                                                    ),
                                                                                                    width={
                                                                                                        "size": 3
                                                                                                    },
                                                                                                ),
                                                                                            ]
                                                                                        ),
                                                                                    ]
                                                                                )
                                                                            )
                                                                        ],
                                                                        style={
                                                                            "display": "inline-block",
                                                                            "margin-top": "-5px",
                                                                        },
                                                                    ),
                                                                ],
                                                                style={
                                                                    "display": "flex",
                                                                    "justify-content": "center",
                                                                },
                                                            ),
                                                        )
                                                    ),
                                                ]
                                            )
                                        )
                                    ]
                                ),
                            ]
                        ),
                    ],
                )
            ),
            style={
                "position": "fixed",
                "top": 0,
                "width": "100%",
                "z-index": 1000,
                "height": "170px",
                "backgroundColor": "white",
            },
        ),
        dbc.Card(
            dbc.CardBody(
                dbc.Row(
                    id="content-container",
                    style={
                        "justifyContent": "center",
                        "alignItems": "center",
                        "height": "100vh",
                        "maxWidth": "1080px",
                        "alignSelf": "center",
                        "margin": "0 auto",
                    },
                )
            ),
            style={"marginTop": "170px"},
        ),
        dcc.Store(id="blogs-df"),
    ]
)