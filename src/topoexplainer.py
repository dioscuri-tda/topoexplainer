import shap
from pyballmapper import BallMapper
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class TopoExplainer:
    def __init__(self, model, data: pd.DataFrame, use_absolute_shap=False):
        self.model = model
        self.data = data
        self.shap_explainer = shap.Explainer(model.predict, data)
        self.shap_values = self.shap_explainer(data)
        if use_absolute_shap:
            self.shap_values.values = np.abs(self.shap_values.values)
        self._shap_values_df = pd.DataFrame(self.shap_values.values, columns=data.columns)
        # _pred values taken from shap. perhaps we should use self.model.predict(data)
        self._model_prediction = self.shap_values.values.sum(axis=1) + self.shap_values.base_values
        self._space = None


    def _average_over_balls(self, values):
        # TODO: check if this is correct
        return [np.mean(values[in_ball_ids]) for in_ball_ids in self.ball_mapper.points_covered_by_landmarks.values()]

    def explain_shap_space(self, eps: float):
        self.ball_mapper = BallMapper(X=self.shap_values.values, eps=eps, verbose=False)
        self._space = 'shap'

    def explain_data_space(self, eps: float):
        self.ball_mapper = BallMapper(X=self.shap_values.data, eps=eps, verbose=False)
        self._space = 'data'

    def plot_explainability_graph(self, feature=None, ax=None, seed=None):
        plot_title = f'Color by feature={feature} importance' if feature is not None else 'Color by prediction value'
        if self._space == 'data':
            if feature is None:
                raise ValueError('Feature must be provided')
            self._in_ball_values = self._average_over_balls(self.data[feature])
            plot_title = f'{plot_title}\n BM graph by data'
        if self._space == 'shap':
            self._in_ball_values = self._average_over_balls(self._model_prediction)
            plot_title = f'{plot_title}\n BM graph by shap values'

        graph_layout = nx.spring_layout(self.ball_mapper.Graph, seed=seed)
        cmap = plt.cm.coolwarm
        norm = mpl.colors.Normalize()
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        node_color = sm.to_rgba(self._in_ball_values)
        show_plot = False
        if ax is None:
            show_plot = True
            fig, ax = plt.subplots(1, 1)
        nx.draw_networkx(self.ball_mapper.Graph, pos=graph_layout, node_color=node_color, ax=ax)
        plt.colorbar(sm, ax=ax)
        ax.set_title(plot_title)
        if show_plot:
            plt.show()