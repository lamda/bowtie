# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals
import random
import io
import pdb

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import prettyplotlib as ppl


# settings
indices = (0, 24, 49, 74, 99)


class Graph(nx.DiGraph):
    def __init__(self, graph=None):
        """Directed graph extended with bow tie functionality
        parameter graph must be a directed networkx graph"""
        super(Graph, self).__init__()
        self.lc_asp, self.lc_diam = 0, 0
        self.bow_tie, self.bow_tie_dict, self.bow_tie_changes = 0, 0, []
        if graph:
            self.add_nodes_from(graph)
            self.add_edges_from(graph.edges())

    def stats(self, prev_bow_tie_dict=None):
        """ calculate several statistical measures on the graphs"""
        # Core, In and Out
        cc = nx.strongly_connected_components(self)
        lc = self.subgraph(cc[0])
        scc = set(lc.nodes())
        scc_node = random.sample(scc, 1)[0]
        sp = nx.all_pairs_shortest_path_length(self)
        inc = {n for n in self.nodes() if scc_node in sp[n]}
        inc -= scc

        outc = set()
        for n in scc:
            outc |= set(sp[n].keys())
        outc -= scc

        # Tendrils, Tube and Other
        tube = set()
        out_tendril = set()
        in_tendril = set()
        other = set()

        remainder = set(self.nodes()) - scc - inc - outc
        inc_out = set()
        for n in inc:
            inc_out |= set(sp[n].keys())
        inc_out = inc_out - inc - scc - outc

        for n in remainder:
            if n in inc_out:
                if set(sp[n].keys()) & outc:
                    tube.add(n)
                else:
                    in_tendril.add(n)
            elif set(sp[n].keys()) & outc:
                out_tendril.add(n)
            else:
                other.add(n)
        self.bow_tie = [inc, scc, outc, in_tendril, out_tendril, tube, other]
        self.bow_tie = [100 * len(x)/len(self) for x in self.bow_tie]
        zipped = zip(['inc', 'scc', 'outc', 'in_tendril', 'out_tendril',
                      'tube', 'other'], range(7))
        c2a = {c: i for c, i in zipped}
        self.bow_tie_dict = {}
        for i, c in enumerate([inc, scc, outc, in_tendril, out_tendril, tube,
                               other]):
            for n in c:
                self.bow_tie_dict[n] = i

        if prev_bow_tie_dict:
            self.bow_tie_changes = np.zeros((len(c2a), len(c2a)))
            for n in self:
                self.bow_tie_changes[prev_bow_tie_dict[n],
                                     self.bow_tie_dict[n]] += 1
            self.bow_tie_changes /= len(self)


class GraphCollection(list):
    def __init__(self, label):
        super(GraphCollection, self).__init__()
        self.label = label

    def compute(self):
        """compute statistics on all the graphs in the collection
        the bow tie changes are only computed between selected indices,
        as indicated by the global variable indices"""
        bow_tie_dict = None
        for i, g in enumerate(self):
            if i in indices:
                g.stats(bow_tie_dict)
                bow_tie_dict = g.bow_tie_dict
            else:
                g.stats()


class Plotting(object):
    def __init__(self, graphs):
        self.graphs = graphs
        self.styles = ['solid', 'dashed']
        self.colors = ppl.set2
        self.stackplot()
        self.alluvial()

    def stackplot(self):
        """produce stackplots for the graphcollections"""
        fig, axes = plt.subplots(1, len(self.graphs), squeeze=False,
                                 figsize=(8 * len(self.graphs), 6))
        legend_proxies = []
        for i, gc in enumerate(self.graphs):
            data = [graph.bow_tie for graph in gc]
            polys = axes[0, i].stackplot(np.arange(1, 1 + len(data)),
                                         np.transpose(np.array(data)),
                                         baseline='zero', edgecolor='face')
            legend_proxies = [plt.Rectangle((0, 0), 1, 1,
                              fc=p.get_facecolor()[0])
                              for p in polys]
        axes[0, -1].legend(legend_proxies, ['IN', 'SCC', 'OUT', 'TL_IN',
                           'TL_OUT', 'TUBE', 'OTHER'],
                           loc='upper center', bbox_to_anchor=(0.5, -0.1),
                           ncol=4, fancybox=True)

        # Beautification
        for col in range(axes.shape[0]):
            for row in range(axes.shape[1]):
                axes[col, row].set_ylabel('% of nodes')
                axes[col, row].set_ylim(0, 100)
                axes[col, row].set_xlim(1, 100)
                axes[col, row].set_xlabel('p in %')

        fig.subplots_adjust(left=0.08, bottom=0.21, right=0.95, top=0.95,
                            wspace=0.25, hspace=0.4)
        # plt.show()
        # save to disk
        fig.savefig('plots/bowtie_stacked.png')
        fig.savefig('plots/bowtie_stacked.pdf')

    def alluvial(self):
        """ produce an alluvial diagram (sort of like a flow chart) for the
        bowtie membership changes for selected indices,
        as indicated by the global variable indices"""
        ind = '    '  # indentation for the printed HTML and JavaScript files
        labels = ['IN', 'SCC', 'OUT', 'TL_IN', 'TL_OUT', 'TUBE', 'OTHER']
        dirpath = 'plots/alluvial/'
        with io.open(dirpath + 'alluvial.html', encoding='utf-8') as infile:
            template = infile.read().split('"data.js"')

        for i, gc in enumerate(self.graphs):
            data = [graph.bow_tie for graph in self.graphs[i]]
            changes = [g.bow_tie_changes
                       for j, g in enumerate(self.graphs[i]) if j in indices]
            fname = 'data_' + gc.label + '.js'
            with io.open(dirpath + fname, 'w', encoding='utf-8') as outfile:
                outfile.write('var data = {\n')
                outfile.write(ind + '"times": [\n')
                for iden, idx in enumerate(indices):
                    t = data[idx]
                    outfile.write(ind * 2 + '[\n')
                    for jdx, n in enumerate(t):
                        outfile.write(ind * 3 + '{\n')
                        outfile.write(ind * 4 + '"nodeName": "Node ' +
                                      unicode(jdx) + '",\n')
                        nid = unicode(iden * len(labels) + jdx)
                        outfile.write(ind * 4 + '"id": ' + nid +
                                      ',\n')
                        outfile.write(ind * 4 + '"nodeValue": ' +
                                      unicode(int(n * 100)) + ',\n')
                        outfile.write(ind * 4 + '"nodeLabel": "' +
                                      labels[jdx] + '"\n')
                        outfile.write(ind * 3 + '}')
                        if jdx != (len(t) - 1):
                            outfile.write(',')
                        outfile.write('\n')
                    outfile.write(ind * 2 + ']')
                    if idx != (len(data) - 1):
                        outfile.write(',')
                    outfile.write('\n')
                outfile.write(ind + '],\n')
                outfile.write(ind + '"links": [\n')

                for cidx, ci in enumerate(changes):
                    for mindex, val in np.ndenumerate(ci):
                        outfile.write(ind * 2 + '{\n')
                        s = unicode((cidx - 1) * len(labels) + mindex[0])
                        t = unicode(cidx * len(labels) + mindex[1])
                        outfile.write(ind * 3 + '"source": ' + s +
                                      ',\n')
                        outfile.write(ind * 3 + '"target": ' + t
                                      + ',\n')
                        outfile.write(ind * 3 + '"value": ' +
                                      unicode(val * 5000) + '\n')
                        outfile.write(ind * 2 + '}')
                        if mindex != (len(ci) - 1):
                            outfile.write(',')
                        outfile.write('\n')
                outfile.write(ind + ']\n')
                outfile.write('}')
            hfname = dirpath + 'alluvial_' + gc.label + '.html'
            with io.open(hfname, 'w', encoding='utf-8') as outfile:
                outfile.write(template[0] + '"' + fname + '"' + template[1])


if __name__ == '__main__':
    # create the graphs
    graphs = []
    gc = GraphCollection('gnp')
    for p in xrange(0, 100):
        g = nx.gnp_random_graph(10, p/100, directed=True)
        gc.append(Graph(g))
    graphs.append(gc)

    # compute statistics
    for g in graphs:
        g.compute()

    # plot
    P = Plotting(graphs)
