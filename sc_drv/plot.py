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


# =============================================================================
# CLASSES
# =============================================================================

class PlotError(ValueError):
    pass


@attr.s(frozen=True)
class PlotProxy(object):

    data = attr.ib()

    def __call__(self, plot="ivr", **kwargs):
        func = getattr(self, plot)
        return func(**kwargs)

    def ivr(self, **kwargs):
        ivrs = self.data.aivr
        n_groups = len(ivrs)
        labels = ["Alt. {}".format(idx) for idx in range(n_groups)]

        if self.data.has_weights:
            n_groups += 1
            ivrs = np.hstack([self.data.wivr, ivrs])
            labels.insert(0, "Weight")

        ax = bar(ivrs, **kwargs)
        ax.axhline(self.data.climit)

        ax.set_ylabel('IVR')
        ax.set_title('IVR vs Consensus limit')
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(labels)

        yticks = np.append(ax.get_yticks(), self.data.climit)
        yticks.sort()
        ax.set_yticks(yticks)
        ax.legend(["climit"])

        return ax

    def consensus(self, **kwargs):
        total = float(len(self.data.ain_consensus))
        count = np.sum(self.data.ain_consensus)

        if self.data.has_weights:
            total += 1
            count += int(self.data.win_consensus)

        trues = count / total
        falses = 1 - trues

        labels = 'Consensus', 'No-Consensus'
        kwargs.setdefault("explode", (0, 0.1))

        ax = pie((trues, falses), labels=labels, **kwargs)
        ax.set_title("Consensus Proportion")

        return ax

    def weights_by_participants(self, **kwargs):
        if not self.data.has_weights:
            raise PlotError("Data without weights")
        ax = box_violin_plot(self.data.weights_participants.T, **kwargs)
        ax.set_xlabel("Participants")
        ax.set_ylabel("Weights")
        ax.set_title("Weights by Participants")
        return ax

    def weights_by_criteria(self, **kwargs):
        if not self.data.has_weights:
            raise PlotError("Data without weights")
        ax = box_violin_plot(self.data.weights_participants, **kwargs)
        ax.set_xlabel("Criteria")
        ax.set_ylabel("Weights")
        ax.set_title("Weights by Criteria")
        return ax

    def utilities_by_participants(self, criterion=None, **kwargs):
        if criterion is None:
            mtx = np.hstack(self.data.mtx_participants).T
            title = "Utilities by Participants - ALL CRITERIA"
        else:
            mtx = self.data.mtx_participants[criterion].T
            title = f"Utilities by Participants - Criterion: {criterion}"

        ax = box_violin_plot(mtx, **kwargs)
        ax.set_xlabel("Participant")
        ax.set_ylabel("Utilities")
        ax.set_title(title)
        return ax

    def utilities_by_alternatives(self, criterion=None, **kwargs):
        if criterion is None:
            mtx = np.vstack(self.data.mtx_participants)
            title = "Utilities by Alternatives - ALL CRITERIA"
        else:
            mtx = self.data.mtx_participants[criterion]
            title = f"Utilities by Alternatives - Criterion: {criterion}"

        ax = box_violin_plot(mtx, **kwargs)
        ax.set_xlabel("Alternatives")
        ax.set_ylabel("Utilities")
        ax.set_title(title)
        return ax
