import ngauge
import numpy as np

def replace_root(neuron):

    # assert all branch heads have the same data
    for branch_n, branch_head in enumerate(neuron.branches):
        if branch_n == 0:
            first_data = np.array([branch_head.x,branch_head.y,branch_head.z,branch_head.r])
        else:
            data = np.array([branch_head.x,branch_head.y,branch_head.z,branch_head.r])
            assert np.array_equal(first_data, data)

    # make a single branch head
    first_root = neuron.branches[0]
    for i in range(1, len(neuron.branches)):
        first_root.children += neuron.branches[i].children

    # change parents of children
    for child in first_root.children:
        child.parent = first_root

    neuron.branches = [first_root]
    return neuron

def resample_neuron(neuron, sampling):
    neuron = replace_root(neuron)
    
    stack = []
    stack += neuron.branches[0].children

    while len(stack) > 0:
        child = stack.pop()
        stack += child.children

        parent = child.parent
        parents_children = parent.children
        for idx, c in enumerate(parents_children):
            if c == child:
                child_idx = idx
                break

        pt1 = np.array([parent.x, parent.y, parent.z])
        pt2 = np.array([child.x, child.y, child.z])

        dist = np.linalg.norm(pt2-pt1)

        if dist > sampling:
            samples = np.arange(sampling, dist, sampling)
            
            for n_sample, sample in enumerate(samples):
                loc = (pt2-pt1)/dist*sample+pt1
                loc = [float(l) for l in loc]
                new_pt = ngauge.TracingPoint(x=loc[0], y=loc[1], z=loc[2], r=1, t=child.t)
                if n_sample == 0: #beginning of chain is loose
                    first_pt = new_pt
                else: # add link to chain
                    new_pt.parent = prev_pt
                    prev_pt.children = [new_pt]

                prev_pt = new_pt
            
            # attach end to child
            child.parent = new_pt
            new_pt.children = [child]

            # attach beginning of chain to parent
            first_pt.parent = parent
            parents_children.pop(child_idx)
            parents_children.append(first_pt)

    return neuron