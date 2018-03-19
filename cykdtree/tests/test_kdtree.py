import numpy as np
import time
import tempfile
from nose.tools import assert_raises
import cykdtree
from cykdtree.tests import parametrize, make_points, make_points_neighbors


@parametrize(npts=100, ndim=(2, 3), periodic=(False, True), 
             use_sliding_midpoint=(False, True))
def test_PyKDTree(npts=100, ndim=2, periodic=False, use_sliding_midpoint=False):
    pts, le, re, ls = make_points(npts, ndim)
    cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic,
                      use_sliding_midpoint=use_sliding_midpoint)


def test_PyKDTree_errors():
    pts, le, re, ls = make_points(100, 2)
    assert_raises(ValueError, cykdtree.PyKDTree, pts, le, re,
                  leafsize=1)


@parametrize(npts=100, ndim=(2, 3), periodic=(False, True))
def test_search(npts=100, ndim=2, periodic=False):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    pos_list = [le, (le+re)/2.]
    if periodic:
        pos_list.append(re)
    for pos in pos_list:
        leaf = tree.get(pos)
        leaf.neighbors


@parametrize(npts=100, ndim=(2, 3))
def test_search_errors(npts=100, ndim=2):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls)
    assert_raises(ValueError, tree.get, re)


@parametrize(periodic=(False, True))
def test_neighbors(periodic=False):
    pts, le, re, ls, left_neighbors, right_neighbors = make_points_neighbors(
        periodic=periodic)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    for leaf in tree.leaves:
        out_str = str(leaf.id)
        try:
            for d in range(tree.ndim):
                out_str += '\nleft:  {} {} {}'.format(d, leaf.left_neighbors[d],
                                               left_neighbors[d][leaf.id])
                assert(len(left_neighbors[d][leaf.id]) ==
                       len(leaf.left_neighbors[d]))
                for i in range(len(leaf.left_neighbors[d])):
                    assert(left_neighbors[d][leaf.id][i] ==
                           leaf.left_neighbors[d][i])
                out_str += '\nright: {} {} {}'.format(d, leaf.right_neighbors[d],
                                                right_neighbors[d][leaf.id])
                assert(len(right_neighbors[d][leaf.id]) ==
                       len(leaf.right_neighbors[d]))
                for i in range(len(leaf.right_neighbors[d])):
                    assert(right_neighbors[d][leaf.id][i] ==
                           leaf.right_neighbors[d][i])
        except:
            for leaf in tree.leaves:
                print(leaf.id, leaf.left_edge, leaf.right_edge)
            print(out_str)
            raise


@parametrize(npts=100, ndim=(2,3), periodic=(False, True))
def test_get_neighbor_ids(npts=100, ndim=2, periodic=False):
    pts, le, re, ls = make_points(npts, ndim)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    pos_list = [le, (le+re)/2.]
    if periodic:
        pos_list.append(re)
    for pos in pos_list:
        tree.get_neighbor_ids(pos)


def time_tree_construction(Ntime, LStime, ndim=2):
    pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
    t0 = time.time()
    cykdtree.PyKDTree(pts, le, re, leafsize=LStime)
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))


def time_neighbor_search(Ntime, LStime, ndim=2):
    pts, le, re, ls = make_points(Ntime, ndim, leafsize=LStime)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=LStime)
    t0 = time.time()
    tree.get_neighbor_ids(0.5*np.ones(tree.ndim, 'double'))
    t1 = time.time()
    print("{} {}D points, leafsize {}: took {} s".format(Ntime, ndim, LStime, t1-t0))

def test_save_load():
    for periodic in (True, False):
        for ndim in range(1, 5):
            pts, le, re, ls = make_points(100, ndim)
            tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                                     periodic=periodic, data_version=ndim+12)
            with tempfile.NamedTemporaryFile() as tf:
                tree.save(tf.name)
                restore_tree = cykdtree.PyKDTree.from_file(tf.name)
                tree.assert_equal(restore_tree)

@parametrize(npts=(10, 100, 1000), periodic=(True, False),
             ndim=list(range(1, 5)), distrib=('rand', 'uniform', 'normal'))
def test_level(npts, periodic, ndim, distrib):
    def level_search(node, level):
        ret = []
        if node.left_child is not None:
            ret.append(level_search(node.left_child, level+1))
        if node.right_child is not None:
            ret.append(level_search(node.right_child, level+1))
        return ret + [node.level == level]

    pts, le, re, ls = make_points(npts, ndim, distrib=distrib)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls, periodic=periodic)
    assert all(level_search(tree.root, 0))

@parametrize(npts=(10, 100, 1000), periodic=(True, False),
             ndim=list(range(1, 5)), distrib=('rand', 'uniform', 'normal'))
def test_amr_nested(npts, periodic, ndim, distrib):
    def nested_search(node):
        if node is None:
            return []
        ret = []
        ret.extend(nested_search(node.left_child))
        ret.extend(nested_search(node.right_child))
        result = True
        if (le == re).all():
            result = False
        if node.npts <= 1:
            result = False
        ret.append(result)
        return ret

    pts, le, re, ls = make_points(npts, ndim, distrib=distrib)
    tree = cykdtree.PyKDTree(pts, le, re, leafsize=ls,
                             periodic=periodic, amr_nested=True)
    assert all(nested_search(tree.root))
