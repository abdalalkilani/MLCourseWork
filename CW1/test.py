top_tree = {'left': { 'left': { 'node': True }, 'right':{ 'node': True }} , 'right': { 'node': True }}
left_node = top_tree['left']

left_node = {'left': {'node': False}}



def update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

update(top_tree, left_node)

print(top_tree)