from pyballmapper import BallMapper
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


class BMLR:
    def __init__(self, eps: float):
        self.eps = eps
        self.fitted = False
        self.ball_mapper = None

    def fit(self, X: pd.DataFrame, y):
        self.data = np.array(X)
        self.columns = X.columns.tolist()
        self.target = y
        self.models_inball = []
        for pt in tqdm(self.data):
            mask = self.find_inball_points_mask(query_point=pt)
            pts_inball = self.data[mask]
            target_inball = self.target[mask]
            inballmodel = LinearRegression().fit(pts_inball, target_inball)
            self.models_inball.append(inballmodel)
        self.fitted = True

    def find_inball_points_mask(self, query_point):
        distances_mask = np.linalg.norm(self.data - query_point, axis=1) < self.eps
        return distances_mask

    def create_graph(self, seed=None):
        order = None
        # if seed is not None:
        #     order = np.random.RandomState(seed=seed).permutation(self.data.shape[0])
        # print(order)
        self.ball_mapper = BallMapper(X=self.data, eps=self.eps, order=order, verbose=False)
        self._graph_layout = nx.spring_layout(self.ball_mapper.Graph, seed=seed)

    def plot_coefficient(self, feature, ax=None, color_map=plt.cm.coolwarm):
        if not self.fitted:
            raise ValueError("Run .fit() first!")
        if self.ball_mapper is None:
            raise ValueError("Run .create_graph() first!")
        if feature not in self.columns:
            raise ValueError(f'Coefficient {feature} not found! Options are {self.columns}')
        column_id = self.columns.index(feature)
        # compute average value of the coeffcient each ball
        node_coef = []
        for in_ball_ids in self.ball_mapper.points_covered_by_landmarks.values():
            avg = np.mean([self.models_inball[pt_id].coef_[column_id] for pt_id in in_ball_ids])
            node_coef.append(avg)
        color_normalization = mpl.colors.Normalize()
        sm = plt.cm.ScalarMappable(norm=color_normalization, cmap=color_map)
        node_color = sm.to_rgba(node_coef)
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            show_plot = True
        nx.draw_networkx(self.ball_mapper.Graph, pos=self._graph_layout, node_color=node_color, ax=ax)
        plt.colorbar(sm, ax=ax)
        ax.set_title(feature)
        if show_plot:
            plt.show()
