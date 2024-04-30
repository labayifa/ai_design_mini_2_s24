# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display': 'grid', 'grid-auto-flow': 'row'}
row_style = {'display': 'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd
import io
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import requests
from request_client import *

API_URL = "http://localhost:6000"

app = Dash(__name__)

df = pd.read_csv("iris_extended_encoded.csv", sep=',')
df_csv = df.to_csv(index=False)

app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
        dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

            html.Div([
                html.Div([
                    html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
                    dcc.Input(id='file-for-train', type='text', style={'width': '100px'}),
                    html.Div([
                        html.Button('Load', id='load-val', style={"width": "60px", "height": "30px"}),
                        html.Div(id='load-response', children='Click to load')
                    ], style=col_style)
                ], style=col_style),

                html.Div([
                    html.Button('Upload', id='upload-val', style={"width": "60px", "height": "30px"}),
                    html.Div(id='upload-response', children='Click to upload')
                ], style=col_style | {'margin-top': '20px'})

            ], style=col_style | {'margin-top': '50px', 'margin-bottom': '50px', 'width': "400px",
                                  'border': '2px solid black'}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Label(['Feature'], style={'font-weight': 'bold'}),
                        dcc.Dropdown(
                            df.columns,  # <dropdown values for histogram>
                            df.columns[0],  # <default value for dropdown>
                            id='hist-column'
                        )
                    ], style=col_style),
                    dcc.Graph(id='selected_hist')
                ], style=col_style | {'height': '400px', 'width': '400px'}),

                html.Div([

                    html.Div([

                        html.Div([
                            html.Label(['X-Axis'], style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                df.columns,  # <dropdown values for scatter plot x-axis>
                                df.columns[0],  # <default value for dropdown>
                                id='xaxis-column'
                            )
                        ]),

                        html.Div([
                            html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
                            dcc.Dropdown(
                                df.columns,  # <dropdown values for scatter plot y-axis>
                                df.columns[1],  # <default value for dropdown>
                                id='yaxis-column'
                            )
                        ])
                    ], style=row_style | {'margin-left': '50px', 'margin-right': '50px'}),

                    dcc.Graph(id='indicator-graphic')
                ], style=col_style)
            ], style=row_style),

            html.Div(id='tablecontainer', children=[
                dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
                                     id='datatable')
            ])
        ]),
        dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style,
                selected_style=tab_selected_style, children=[
                html.Div([
                    html.Div([
                        html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                        html.Div(dcc.Input(id='dataset-for-train', type='text'))
                    ], style=col_style | {'margin-top': '20px'}),

                    html.Div([
                        html.Button('New model', id='build-val', style={'width': '90px', "height": "30px"}),
                        html.Div(id='build-response', children='Click to build new model and train')
                    ], style=col_style | {'margin-top': '20px'}),

                    html.Div([
                        html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                        html.Div(dcc.Input(id='model-for-train', type='text'))
                    ], style=col_style | {'margin-top': '20px'}),

                    html.Div([
                        html.Button('Re-Train', id='train-val', style={"width": "90px", "height": "30px"})
                    ], style=col_style | {'margin-top': '20px', 'width': '90px'})

                ], style=col_style | {'margin-top': '50px', 'margin-bottom': '50px', 'width': "400px",
                                      'border': '2px solid black'}),

                html.Div(id='container-button-train', children='')
            ]),
        dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.Div([
                    html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                    html.Div(dcc.Input(id='row-for-score', type='text', style={'width': '300px'}))
                ], style=col_style | {'margin-top': '20px'}),
                html.Div([
                    html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                    html.Div(dcc.Input(id='model-for-score', type='text'))
                ], style=col_style | {'margin-top': '20px'}),
                html.Div([
                    html.Button('Score', id='score-val', style={'width': '90px', "height": "30px"}),
                    html.Div(id='score-response', children='Click to score')
                ], style=col_style | {'margin-top': '20px'})
            ], style=col_style | {'margin-top': '50px', 'margin-bottom': '50px', 'width': "400px",
                                  'border': '2px solid black'}),

            html.Div(id='container-button-score', children='')
        ]),

        dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
            html.Div([
                html.Div([
                    html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                    html.Div(dcc.Input(id='dataset-for-test', type='text'))
                ], style=col_style | {'margin-top': '20px'}),
                html.Div([
                    html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                    html.Div(dcc.Input(id='model-for-test', type='text'))
                ], style=col_style | {'margin-top': '20px'}),

                html.Div([
                    html.Button('Test', id='test-val'),
                ], style=col_style | {'margin-top': '20px', 'width': '90px'})

            ], style=col_style | {'margin-top': '50px', 'margin-bottom': '50px', 'width': "400px",
                                  'border': '2px solid black'}),

            html.Div(id='container-button-test', children='')
        ])

    ])
])


# callbacks for Explore data tab

@app.callback(
    # callback annotations go here
    Output(component_id='load-response', component_property='children'),
    Input(component_id='load-val', component_property='n_clicks'),
    State(component_id='file-for-train', component_property='value')
)
def update_output_load(nclicks, filename):
    global df, df_csv

    if nclicks != None:
        print("Hello test")
        # load local data given input filename
        if filename is None or filename == '':
            return 'Load failed.'
        try:
            df = pd.read_csv(filename, sep=',')
            df_csv = df.to_csv(index=False)
        except FileNotFoundError:
            return 'Load failed: File not found.'
        except Exception as e:
            return f'Load failed: {str(e)}'
        return 'Load done.'
    else:
        return ''


@app.callback(
    # callback annotations go here
    Output(component_id='build-response', component_property='children'),
    Input(component_id='build-val', component_property='n_clicks'),
    State(component_id='dataset-for-train', component_property='value')
)
def update_output_build(nclicks, dataset_id):
    print(nclicks)
    if nclicks != None:
        # invoke new model endpoint to build and train model given data set ID
        model_id = -1
        if dataset_id is None or dataset_id == '':
            return '-1'
        try:
            dataset_id = int(dataset_id)
            print("Creating a new model with the uploaded dataset.")
            model_resp = create_iris_model(dataset_id)
            print("Model created response: ", model_resp.text)
            model_id = model_resp.text
        except FileNotFoundError:
            return 'Load failed: File not found.'
        except Exception as e:
            return f'Load failed: {str(e)}'
        # return the model ID
        return model_id
    else:
        return ''


@app.callback(
    # callback annotations go here
    Output(component_id='upload-response', component_property='children'),
    Input(component_id='upload-val', component_property='n_clicks')
)
def update_output_upload(nclicks):
    global df_csv

    if nclicks is not None:
        # invoke the upload API endpoint
        try:
            upload_resp = upload_iris_dataset(f"{df_csv}")
            if upload_resp.status_code == 201:
                # return the dataset ID generated
                return upload_resp.text
            return 'Load Failed : Service unavailable.'
        except FileNotFoundError:
            return 'Load failed: File not found.'
        except Exception as e:
            return f'Load failed: {str(e)}'
    else:
        return ''


@app.callback(
    # callback annotations go here  yaxis-column
    Output(component_id='indicator-graphic', component_property='figure'),
    Input(component_id='xaxis-column', component_property='value'),
    Input(component_id='yaxis-column', component_property='value')
)
def update_graph(xaxis_column_name, yaxis_column_name):
    fig = px.scatter(x=df.loc[:, xaxis_column_name].values,
                     y=df.loc[:, yaxis_column_name].values)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
    # callback annotations go here
    Output(component_id='selected_hist', component_property='figure'),
    Input(component_id='hist-column', component_property='value')
)
def update_hist(hist_column_name):
    fig = px.histogram(df, x=hist_column_name, nbins=20)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig


@app.callback(
    # callback annotations go here
    Output(component_id='tablecontainer', component_property='children'),
    Input(component_id='load-val', component_property='n_clicks'),

)
def update_table(nclicks):
    return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
                                id='datatable')


# callbacks for Training tab
@app.callback(
    # callback annotations go here
    Output(component_id='container-button-train', component_property='children'),
    Input(component_id='train-val', component_property='n_clicks'),
    State(component_id='model-for-train', component_property='value'),
    State(component_id='dataset-for-train', component_property='value')
)
def update_output_train(nclicks, model_id, dataset_id):
    if nclicks is not None:
        # add API endpoint request here
        model_id = int(model_id)
        print("Retraining the model with the uploaded dataset.")
        retrain_resp = retrain_iris_model(model_id, dataset_id)
        print("Retrained model response: ", retrain_resp.text)
        # api_response_io = retrain_resp.text
        # train_df = pd.read_json(api_response_io)
        # train_fig = px.line(train_df)

        return retrain_resp.text
    else:
        return ""


# callbacks for Scoring tab

@app.callback(
    # callback annotations go here
    Output(component_id='score-response', component_property='children'),
    Input(component_id='score-val', component_property='n_clicks'),
    State(component_id='row-for-score', component_property='value'),
    State(component_id='model-for-score', component_property='value')
)
def update_output_score(nclicks, row_data, model_id):
    if nclicks != None:
        print("Scoring a selected single row.", row_data, model_id)
        # add API endpoint request for scoring here with constructed input row
        score_result = get_score_model(int(model_id), row_data)
        print("Score response: ", score_result.text)
        return score_result.text
    else:
        return ""


# callbacks for Testing tab

@app.callback(
    # callback annotations go here
    Output(component_id='container-button-test', component_property='children'),
    Input(component_id='test-val', component_property='n_clicks'),
    State(component_id='dataset-for-test', component_property='value'),
    State(component_id='model-for-test', component_property='value')
)
def update_output_test(nclicks, dataset_id, model_id):
    if nclicks != None:
        # add API endpoint request for testing with given dataset ID
        r = test_model_data(int(model_id), int(dataset_id))
        api_response_io = io.StringIO(r.text)
        test_df = pd.read_json(api_response_io)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Confusion Matrix", "Precision and Recall Scores"))

        # Add confusion matrix plot
        confusion_matrix_trace = go.Heatmap(z=test_df["confusion_matrix"], colorscale='Viridis')
        fig.add_trace(confusion_matrix_trace, row=1, col=1)

        # Add precision and recall scores plot
        precision_recall_trace = go.Scatter(x=test_df.index, y=test_df["precision_score"], mode='lines',
                                            name='Precision Score')
        fig.add_trace(precision_recall_trace, row=1, col=2)
        recall_trace = go.Scatter(x=test_df.index, y=test_df["recall_score"], mode='lines', name='Recall Score')
        fig.add_trace(recall_trace, row=1, col=2)

        # Update layout
        fig.update_layout(height=600, width=1000, title_text="Confusion Matrix and Precision-Recall Scores")

        # Return dcc.Graph with the combined figure
        return dcc.Graph(figure=fig)
    else:
        return ""


'''  STARTING OF MULTI_LINE COMMENT FIELD...move code below above triple quotes to fill in and run

'''

if __name__ == '__main__':
    app.run_server(debug=True)
