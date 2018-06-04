#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, Cabral, Juan; Luczywo, Nadia; Zanazi Jose Luis
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# =============================================================================
# DOCS
# =============================================================================

"""Plotting routines

"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from matplotlib import cm, pyplot as plt

import attr

from .libs import heatmap as plthmap


# =============================================================================
# FUNCTION
# =============================================================================

def box_violin_plot(mtx, ptype="box", cmap=None, ax=None,
                    subplots_kwargs=None, plot_kwargs=None):
    # create ax if necesary
    if ax is None:
        subplots_kwargs = subplots_kwargs or {}
        ax = plt.subplots(**subplots_kwargs)[-1]

    # plot creation
    plot_kwargs = plot_kwargs or {}
    if ptype == "violin":
        key = "bodies"
        plot = ax.violinplot(mtx, **plot_kwargs)
    elif ptype == "box":
        key = "boxes"
        plot_kwargs.setdefault("notch", False)
        plot_kwargs.setdefault("vert", True)
        plot_kwargs.setdefault("patch_artist", True)
        plot_kwargs.setdefault("sym", "o")
        plot_kwargs.setdefault("flierprops", {'linestyle': 'none',
                                              'marker': 'o',
                                              'markerfacecolor': 'red'})
        plot = ax.boxplot(mtx, **plot_kwargs)
    else:
        raise ValueError("ptype must be 'box' or 'violin'")

    # colors in boxes
    cmap = cm.get_cmap(name=cmap)
    colors = cmap(np.linspace(0.35, 0.8, mtx.shape[1]))
    for box, color in zip(plot[key], colors):
        box.set_facecolor(color)
    ax.get_figure().tight_layout()
    return ax


def pie(sizes, explode=None, labels=None, cmap=None, ax=None,
        subplots_kwargs=None, plot_kwargs=None):
            # create ax if necesary
            if ax is None:
                subplots_kwargs = subplots_kwargs or {}
                ax = plt.subplots(**subplots_kwargs)[-1]

            if explode is None:
                explode = [0] * len(sizes)
            if labels is None:
                labels = ["Data {}".format(idx) for idx in range(len(sizes))]

            plot_kwargs = plot_kwargs or {}
            plot_kwargs.setdefault("autopct", '%1.1f%%')
            plot_kwargs.setdefault("shadow", True)
            plot_kwargs.setdefault("startangle", 90)

            labels = [
                l if sizes[idx] > 0 else "" for idx, l in enumerate(labels)]
            plot = ax.pie(sizes, explode=explode, labels=labels, **plot_kwargs)

            # colors in slides
            cmap = cm.get_cmap(name=cmap)
            colors = cmap(np.linspace(0.35, 0.8, len(sizes)))
            for wedge, color in zip(plot[0], colors):
                wedge.set_facecolor(color)

            ax.axis('equal')
            ax.get_figure().tight_layout()

            return ax


def bar(values, cmap=None, ax=None,
        subplots_kwargs=None, plot_kwargs=None):

            # create ax if necesary
            if ax is None:
                subplots_kwargs = subplots_kwargs or {}
                ax = plt.subplots(**subplots_kwargs)[-1]

            plot_kwargs = plot_kwargs or {}
            plot_kwargs.setdefault("width", 0.35)
            plot_kwargs.setdefault("alpha", 0.4)

            # colors in bars
            idxs = np.arange(len(values))
            cmap = cm.get_cmap(name=cmap)
            colors = cmap(np.linspace(0.35, 0.8, len(values)))
            for idx, val, color in zip(idxs, values, colors):
                ax.bar(idx, val, color=color, **plot_kwargs)
            ax.get_figure().tight_layout()
            return ax


def annotated_heatmap(values, row_labels, col_labels, cbar_label, cmap=None,
                      ax=None, subplots_kwargs=None, plot_kwargs=None):

    # create ax if necesary
    if ax is None:
        subplots_kwargs = subplots_kwargs or {}
        ax = plt.subplots(**subplots_kwargs)[-1]

    plot_kwargs = plot_kwargs or {}
    im, cbar = plthmap.heatmap(
        data=values,
        row_labels=row_labels,
        col_labels=col_labels,
        cbarlabel=cbar_label,
        ax=ax, cmap=cmap,
        **plot_kwargs)

    plthmap.annotate_heatmap(im)
    ax.get_figure().tight_layout()

    return ax


# =============================================================================
# CLASSES
# =============================================================================

class PlotError(ValueError):
    pass


@attr.s(frozen=True)
class PlotProxy(object):
    """DRVResult plotting accessor and method

    Examples
    --------
    >>> result.plot.consensus()
    >>> df.plot.irv()

    These plotting methods can also be accessed by calling the accessor as a
    method with the ``kind`` argument:
    ``result.plot(kind='irv')`` is equivalent to ``df.plot.irv()``

    """

    data = attr.ib()

    def __call__(self, kind="ivr", **kwargs):
        func = getattr(self, kind)
        return func(**kwargs)

    def ivr(self, **kwargs):
        """Consensus level achieved by the subproblems as a barchart.

        The subproblems with consensus are lower than the Consensus Limit
        (climit).

        Parameters
        ----------

        cmap: str or None, default: None
            Color map. If is None the default colormap is used.

        ax: Axis

        subplots_kwargs : dict or None

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.bar` function.

        """
        ivrs = self.data.aivr_
        n_groups = len(ivrs)
        labels = ["$j_{}$".format(idx) for idx in range(n_groups)]

        if self.data.has_weights_:
            n_groups += 1
            ivrs = np.hstack([self.data.wivr_, ivrs])
            labels.insert(0, "$j_{W}$")

        ax = bar(ivrs, **kwargs)
        ax.axhline(self.data.climit)

        ax.set_ylabel('IVR')
        ax.set_title('IVR vs Consensus limit')

        ax.set_xlabel("Subproblems")
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(labels)

        yticks = np.append(ax.get_yticks(), self.data.climit)
        yticks.sort()
        ax.set_yticks(yticks)
        ax.legend(["climit"])

        return ax

    def consensus(self, **kwargs):
        """Piechart with the proportion subproblem with and without consensus.

        Parameters
        ----------

        explode: tuple of length 2, optional (default=(0, 0.1))
            Separation between the slides.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.pie` function.

        """
        total = float(len(self.data.ain_consensus_))
        count = np.sum(self.data.ain_consensus_)

        if self.data.has_weights_:
            total += 1
            count += int(self.data.win_consensus_)

        trues = count / total
        falses = 1 - trues

        labels = 'Consensus', 'No-Consensus'
        kwargs.setdefault("explode", (0, 0.1))

        ax = pie((trues, falses), labels=labels, **kwargs)
        ax.set_title("Consensus Proportion")

        return ax

    def weight_heatmap(self, **kwargs):
        """Create a heat-map matrix of the selected weight.

        Parameters
        ----------

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.imshow` function.

        """
        if not self.data.has_weights_:
            raise PlotError("Data without weights")

        row_labels = [
            "$n_{}$".format(idx) for idx in range(self.data.N_)]
        col_labels = [
            "$j_{}$".format(idx) for idx in range(self.data.J_)]
        data = self.data.wmtx_

        ax = annotated_heatmap(
            values=data, row_labels=row_labels, col_labels=col_labels,
            cbar_label="Weights", **kwargs)

        ax.set_ylabel("Participants")
        ax.set_xlabel("Subproblems")
        ax.set_title("Weights", y=1.15)

        return ax

    def weights_by_participants(self, **kwargs):
        """Distribution of weigths of criteria by participant.

        Parameters
        ----------

        ptype : {"box", "violin"}, optional (default="box")
            The plot type.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.box` or
            `maptplotlib.pyplot.violin` function.

        """
        if not self.data.has_weights_:
            raise PlotError("Data without weights")
        ax = box_violin_plot(self.data.wmtx_.T, **kwargs)

        ax.set_xticks(np.arange(self.data.N_) + 1)
        ax.set_xticklabels([
            "$n_{}$".format(idx) for idx in range(self.data.N_)])

        ax.set_xlabel("Participants")
        ax.set_ylabel("Weights")
        ax.set_title("Weights by Participants")
        return ax

    def weights_by_subproblems(self, **kwargs):
        """Distribution of weigths of criteria by subproblem.

        Parameters
        ----------

        ptype : {"box", "violin"}, optional (default="box")
            The plot type.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.box` or
            `maptplotlib.pyplot.violin` function.

        """
        if not self.data.has_weights_:
            raise PlotError("Data without weights")
        ax = box_violin_plot(self.data.wmtx_, **kwargs)

        ax.set_xticks(np.arange(self.data.J_) + 1)
        ax.set_xticklabels([
            "$j_{}$".format(idx) for idx in range(self.data.J_)])

        ax.set_xlabel("Subproblem")
        ax.set_title("Weights by Criteria")
        return ax

    def utilities_heatmap(self, subproblem=None, **kwargs):
        """Create a heatmap matrix of the selected utilities.

        Parameters
        ----------

        subproblem : int or None.
            If a subproblem number is provided only the utilities of
            this subproblem is shown. Otherwise the utilities are added
            toguether in a single plot.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.imshow` function.

        """
        if subproblem is None:
            data = np.add.reduce(self.data.amtx_criteria_)
            title = "Utilities - ALL SUBPROBLEMS"
        else:
            data = self.data.amtx_criteria_[subproblem]
            title = f"Utilities - Subproblem: $J_{subproblem}$"

        col_labels = [
            "$i_{}$".format(idx) for idx in range(self.data.I_)]
        row_labels = [
            "$n_{}$".format(idx) for idx in range(self.data.N_)]

        ax = annotated_heatmap(
            values=data, row_labels=row_labels, col_labels=col_labels,
            cbar_label="Utilities", **kwargs)

        ax.set_ylabel("Participants")
        ax.set_xlabel("Alternatives")
        ax.set_title(title, y=1.15)
        return ax

    def utilities_by_participants(self, subproblem=None, **kwargs):
        """Distribution of selected utilities by participants.

        Parameters
        ----------

        subproblem : int or None, optional (default=None)
            If a subproblem number is provided only the utilities of
            this subproblem is shown. Otherwise the utilities are added
            toguether in a single plot.

        ptype : {"box", "violin"}, optional (default="box")
            The plot type.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.box` or
            `maptplotlib.pyplot.violin` function.

        """
        if subproblem is None:
            mtx = np.hstack(self.data.amtx_criteria_).T
            title = "Utilities by Participants - ALL SUBPROBLEMS"
        else:
            mtx = self.data.amtx_criteria_[subproblem].T
            title = f"Utilities by Participants - Subproblem: $J_{subproblem}$"

        ax = box_violin_plot(mtx, **kwargs)

        ax.set_xticks(np.arange(self.data.N_) + 1)
        ax.set_xticklabels([
            "$n_{}$".format(idx) for idx in range(self.data.N_)])

        ax.set_xlabel("Participants")
        ax.set_ylabel("Utilities")
        ax.set_title(title)
        return ax

    def utilities_by_alternatives(self, subproblem=None, **kwargs):
        """Distribution of selected utilities by alternatives.

        Parameters
        ----------

        subproblem : int or None, optional (default=None)
            If a subproblem number is provided only the utilities of
            this subproblem is shown. Otherwise the utilities are added
            toguether in a single plot.

        ptype : {"box", "violin"}, optional (default="box")
            The plot type.

        cmap: str, optional (default: None)
            Color map. If is None the default colormap is used.

        ax: matplotlib axes object, default None.

        subplots_kwargs : dict or None
            Parameters to the subplot function if no axis is provided.

        plot_kwargs : dict or None
            Parameters of the `maptplotlib.pyplot.box` or
            `maptplotlib.pyplot.violin` function.

        """
        if subproblem is None:
            mtx = np.vstack(self.data.amtx_criteria_)
            title = "Utilities by Alternatives - ALL SUBPROBLEMS"
        else:
            mtx = self.data.amtx_criteria_[subproblem]
            title = f"Utilities by Alternatives - Subproblem: $J_{subproblem}$"

        ax = box_violin_plot(mtx, **kwargs)

        ax.set_xticks(np.arange(self.data.I_) + 1)
        ax.set_xticklabels([
            "$i_{}$".format(idx) for idx in range(self.data.I_)])

        ax.set_xlabel("Alternatives")
        ax.set_ylabel("Utilities")
        ax.set_title(title)
        return ax
