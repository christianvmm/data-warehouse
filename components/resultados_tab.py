from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.figure_factory as ff

from utils.read_file_for_resultados import read_file_for_resultados

# Variable global
global_df = pd.DataFrame()


def resultados_tab(filepath):
    return html.Div("Toma de decisión: Visualizar el objetivo y las gráficas que demuestran ese dato para la toma de decisiones.")
